"""
Safety Evaluator Module

Consumes:
 - detections from YOLO / ensembles
 - uncertainty score
 - consensus score
 - optional altitude

Outputs:
 - safety score
 - SAFE / CAUTION / ABORT decision
 - explanation string
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque

# ============================================================
# LOGGING SETUP
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SafetyEvaluator")

# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class SafetyConfig:

    safety_distance_meters: Dict[str, float] = field(default_factory=lambda: {
        'Person': 10.0,
        'Vehicle': 15.0
    })

    safety_radius_pixels: Dict[str, int] = field(default_factory=lambda: {
        'Person': 50,
        'Vehicle': 80
    })

    blur_threshold: float = 100.0
    edge_margin: float = 0.15
    texture_variance_max: float = 50.0
    min_pad_area_ratio: float = 0.002

    safe_threshold: float = 0.65
    caution_threshold: float = 0.45
    max_cumulative_penalty: float = 0.5

    min_obstacle_confidence: float = 0.3
    min_pad_confidence: float = 0.2

    uncertainty_warning: float = 0.3
    consensus_warning: float = 0.7

    min_brightness: float = 30.0
    max_brightness: float = 240.0

    altitude_meters: Optional[float] = None
    camera_hfov_degrees: float = 82.0

    # Penalty weights
    blur_penalty_weight: float = 0.20
    uncertainty_penalty_weight: float = 0.30
    consensus_penalty_weight: float = 0.30
    edge_penalty: float = 0.15
    terrain_penalty_weight: float = 0.10


# ============================================================
# SAFETY EVALUATOR
# ============================================================

class SafetyEvaluator:

    CLASS_NAMES = {0: 'Vehicle', 1: 'UAP', 2: 'UAI', 3: 'Person'}

    PAD_CLASS_IDS = {1, 2}
    OBSTACLE_CLASS_IDS = {0, 3}

    def __init__(self, config: Optional[SafetyConfig] = None):

        self.config = config or SafetyConfig()
        self.score_history = deque(maxlen=10)

        logger.info("SafetyEvaluator initialized")
        logger.info("CONFIG SNAPSHOT:\n%s", self.config)

    # ========================================================
    # MAIN ENTRY
    # ========================================================

    def assess_landing_safety(
        self,
        image_path: str,
        detections: List[Dict],
        uncertainty: float = 0.0,
        consensus: float = 1.0,
        altitude_meters: Optional[float] = None
    ) -> Dict:

        image = self._load_image(image_path)
        if image is None:
            return self._abort_result("Failed to load image", "image_load_failure")

        h, w = image.shape[:2]

        pad_detections = [
            det for det in detections
            if det['class_id'] in self.PAD_CLASS_IDS
            and self._get_confidence(det) >= self.config.min_pad_confidence
        ]

        if not pad_detections:
            return self._abort_result("No landing pad detected", "no_pad_detected")

        best_result = None

        for pad in pad_detections:
            result = self._evaluate_single_pad(
                image, pad, detections,
                uncertainty, consensus,
                altitude_meters, (h, w)
            )

            if best_result is None or result['safety_score'] > best_result['safety_score']:
                best_result = result

        return best_result

    # ========================================================
    # PER PAD EVALUATION
    # ========================================================

    def _evaluate_single_pad(
        self,
        image,
        pad_detection,
        all_detections,
        uncertainty,
        consensus,
        altitude_meters,
        img_size
    ):

        h, w = img_size
        risk_factors = []
        penalties = []

        base_conf = self._get_confidence(pad_detection)
        pad_bbox = self._ensure_numpy(pad_detection['bbox'])
        pad_center = self._get_bbox_center(pad_bbox)

        # ---- PAD SIZE ----
        pad_ratio = self._bbox_area(pad_bbox) / (h * w)
        if pad_ratio < self.config.min_pad_area_ratio:
            penalties.append(0.3)
            risk_factors.append(self._rf("pad_too_small", 0.3, "high"))

        # ---- OBSTACLES ----
        for det in all_detections:

            if det['class_id'] not in self.OBSTACLE_CLASS_IDS:
                continue

            if self._get_confidence(det) < self.config.min_obstacle_confidence:
                continue

            obj_bbox = self._ensure_numpy(det['bbox'])
            obj_center = self._get_bbox_center(obj_bbox)

            dist_px = self._euclidean_distance(pad_center, obj_center)

            safe_px = self._get_safe_distance(
                self.CLASS_NAMES[det['class_id']],
                altitude_meters, w
            )

            if dist_px < safe_px:
                ratio = dist_px / safe_px
                penalty = (1 - ratio) * 0.4 * self._get_confidence(det)

                penalties.append(penalty)
                risk_factors.append(
                    self._rf("obstacle_proximity", penalty, "critical")
                )

        # ---- BLUR ----
        pad_patch = self._extract_bbox_patch_safe(image, pad_bbox, img_size)

        blur = self._detect_motion_blur(pad_patch if pad_patch is not None else image)

        if blur < self.config.blur_threshold:
            penalty = self.config.blur_penalty_weight * (
                1 - blur / self.config.blur_threshold
            )
            penalties.append(penalty)
            risk_factors.append(self._rf("motion_blur", penalty, "medium"))

        # ---- EDGE ----
        if self._is_near_edge(pad_bbox, img_size):
            penalties.append(self.config.edge_penalty)
            risk_factors.append(self._rf("edge_proximity", self.config.edge_penalty, "medium"))

        # ---- UNCERTAINTY ----
        if uncertainty > self.config.uncertainty_warning:
            penalty = uncertainty * self.config.uncertainty_penalty_weight
            penalties.append(penalty)
            risk_factors.append(self._rf("high_uncertainty", penalty, "high"))

        # ---- CONSENSUS ----
        if consensus < self.config.consensus_warning:
            penalty = (1 - consensus) * self.config.consensus_penalty_weight
            penalties.append(penalty)
            risk_factors.append(self._rf("low_consensus", penalty, "high"))

        # ---- TERRAIN ----
        if pad_patch is not None:
            texture = self._compute_texture_variance(pad_patch)
            if texture > self.config.texture_variance_max:
                penalty = self.config.terrain_penalty_weight
                penalties.append(penalty)
                risk_factors.append(self._rf("complex_terrain", penalty, "low"))

        # ====================================================
        # FINAL SCORE (TEMPORAL SMOOTHED)
        # ====================================================

        total_penalty = min(sum(penalties), self.config.max_cumulative_penalty)

        raw_score = base_conf * (1 - total_penalty)

        self.score_history.append(raw_score)
        smoothed_score = float(np.mean(self.score_history))

        safety_score = np.clip(smoothed_score, 0, 1)

        if safety_score >= self.config.safe_threshold:
            decision = "SAFE"
        elif safety_score >= self.config.caution_threshold:
            decision = "CAUTION"
        else:
            decision = "ABORT"

        explanation = self._generate_explanation(risk_factors, decision)

        return {
            "safety_score": round(float(safety_score), 4),
            "decision": decision,
            "risk_factors": risk_factors,
            "explanation": explanation,
            "uncertainty": uncertainty,
            "consensus": consensus,
            "pad_detection": pad_detection
        }

    # ========================================================
    # HELPERS
    # ========================================================

    def _rf(self, t, p, sev):
        return {"type": t, "penalty": round(p, 3), "severity": sev}

    def _generate_explanation(self, risks, decision):

        if not risks:
            return "Landing zone clear. All safety checks passed."

        top = sorted(risks, key=lambda r: r["penalty"], reverse=True)[:3]
        causes = ", ".join(r["type"] for r in top)

        return f"{decision}: primary risks = {causes}"

    def _load_image(self, p):

        img = cv2.imread(p)
        if img is None or img.size == 0:
            return None
        return img

    def _get_confidence(self, det):

        return det.get("calibrated_confidence", det.get("confidence", 0.0))

    def _ensure_numpy(self, b):

        return np.array(b, dtype=float)

    def _bbox_area(self, b):

        return max(0, b[2] - b[0]) * max(0, b[3] - b[1])

    def _get_bbox_center(self, b):

        return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)

    def _euclidean_distance(self, a, b):

        return float(np.linalg.norm(np.array(a) - np.array(b)))

    def _get_safe_distance(self, obj, altitude, img_w):

        alt = altitude or self.config.altitude_meters

        if alt:

            hfov = np.radians(self.config.camera_hfov_degrees / 2)
            ground_w = 2 * alt * np.tan(hfov)
            px_per_m = img_w / ground_w

            meters = self.config.safety_distance_meters.get(obj, 10)

            safe_px = meters * px_per_m

            return float(np.clip(safe_px, 20, img_w * 0.5))

        return float(self.config.safety_radius_pixels.get(obj, 50))

    def _detect_motion_blur(self, img):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _is_near_edge(self, b, size):

        h, w = size
        return (
            b[0] < w * self.config.edge_margin
            or b[1] < h * self.config.edge_margin
            or b[2] > w * (1 - self.config.edge_margin)
            or b[3] > h * (1 - self.config.edge_margin)
        )

    def _extract_bbox_patch_safe(self, img, b, size):

        h, w = size
        x1, y1 = max(0, int(b[0])), max(0, int(b[1]))
        x2, y2 = min(w, int(b[2])), min(h, int(b[3]))

        if x2 <= x1 or y2 <= y1:
            return None

        patch = img[y1:y2, x1:x2]
        return patch if patch.size else None

    def _compute_texture_variance(self, patch):

        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        return float(np.std(gray))

    # --------------------------------------------------
    # INTERNAL ABORT HANDLER
    # --------------------------------------------------

    def _abort_result(self, reason: str, code: str):
        """
        Return standardized abort dictionary when landing is unsafe.
        """

        return {
            "safety_score": 0.0,
            "decision": "ABORT",
            "reason": reason,
            "risk_factors": [code],
            "pad_detection": None,
            "uncertainty": None,
            "consensus": None,
        }



# ============================================================
# END OF FILE
# ============================================================
