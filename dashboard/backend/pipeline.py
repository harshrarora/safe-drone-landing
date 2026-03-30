"""
MC Dropout Pipeline
==================

End-to-end landing safety assessment pipeline using:

- YOLO11
- Monte Carlo Dropout for uncertainty
- Temperature Scaling calibration
- Risk Assessment logic

Works WITHOUT ensemble models.
"""

import sys
from pathlib import Path
import json
import cv2
import numpy as np

# Allow project-root imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from inference.uncertainty.mc_dropout import MCDropoutYOLO
from inference.calibration.temperature_scaling import TemperatureScaling
from inference.risk_assessment.safety_evaluator import SafetyEvaluator


# ---------------------------------------------------------
# JSON UTILITY
# ---------------------------------------------------------

def make_json_safe(obj):
    """Recursively convert numpy types to Python native types."""

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)

    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)

    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]

    return obj


# ---------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------

class MCDropoutPipeline:
    """
    Full inference pipeline using MC Dropout uncertainty.
    """

    def __init__(
        self,
        model_path="models/yolo/augmented_v1/best.pt",
        temperature_path="models/calibration/temperature_augmented.pkl",
    ):
        print("\n🚀 Initializing MC Dropout Pipeline...")

        # --- MC Dropout ---
        self.mc_dropout = MCDropoutYOLO(
            model_path=model_path,
            n_iterations=30,
            dropout_rate=0.2,
        )

        # --- Calibration ---
        self.calibrator = TemperatureScaling(model_path)
        self.calibrator.load_temperature(temperature_path)

        # --- Risk Assessment ---
        self.safety_evaluator = SafetyEvaluator()

        print("✅ Pipeline initialized successfully!")

    # -----------------------------------------------------

    def process(self, image_path: str) -> dict:

        print("\n" + "=" * 70)
        print(f"🛰️  Processing image: {image_path}")

        # -------------------------------------------------
        # 1. MC DROPOUT
        # -------------------------------------------------

        print("\n[1/3] Running MC Dropout uncertainty estimation...")

        mc_result = self.mc_dropout.predict_with_uncertainty(image_path)

        consensus = mc_result["consensus"]
        bbox_uncertainty = mc_result["bbox_uncertainty"]

        print(f"   → Consensus:        {consensus:.2%}")
        print(f"   → BBox uncertainty: {bbox_uncertainty:.4f}")

        # -------------------------------------------------
        # 2. CALIBRATION
        # -------------------------------------------------

        print("\n[2/3] Applying temperature scaling calibration...")

        cal_result = self.calibrator.predict_calibrated(image_path)
        detections = cal_result["detections"]

        print(f"   → {len(detections)} detections")

        for det in detections[:3]:
            print(
                f"     Class {det['class_id']} | "
                f"{det['raw_confidence']:.3f} → "
                f"{det['calibrated_confidence']:.3f}"
            )

        # -------------------------------------------------
        # 3. SAFETY ASSESSMENT  (SAFE WRAPPER)
        # -------------------------------------------------

        print("\n[3/3] Running safety evaluator...")

        try:
            safety = self.safety_evaluator.assess_landing_safety(
                image_path=image_path,
                detections=detections,
                uncertainty=bbox_uncertainty,
                consensus=consensus,
            )

        except Exception as e:
            print("⚠️ Safety evaluator failed → forcing ABORT")
            print("   Error:", e)

            safety = {
                "safety_score": 0.0,
                "decision": "ABORT",
                "reason": "safety_module_error",
                "risk_factors": ["safety_module_error"],
            }

        print("\n🛑 FINAL DECISION")
        print("-" * 40)
        print(f"Decision:     {safety['decision']}")
        print(f"Safety Score: {safety['safety_score']:.3f}")
        print(f"Risk Factors: {len(safety.get('risk_factors', []))}")

        return {
            "image_path": image_path,
            "mode": "mc_dropout",
            "uncertainty": {
                "method": "mc_dropout",
                "consensus": consensus,
                "bbox_uncertainty": bbox_uncertainty,
            },
            "detections": detections,
            "safety_assessment": safety,
            "final_decision": safety["decision"],
            "final_score": safety["safety_score"],
        }

    # -----------------------------------------------------

    def process_batch(self, image_folder: str, max_images=10):

        images = list(Path(image_folder).glob("*.jpg")) + \
                 list(Path(image_folder).glob("*.png"))

        print(f"\n📂 Found {len(images)} images")

        results = []

        for img in images[:max_images]:
            try:
                res = self.process(str(img))
                results.append(res)
            except Exception as e:
                print(f"❌ Failed on {img}: {e}")

        return results

    # -----------------------------------------------------

    def save_visualization(self, result: dict):

        image_path = result["image_path"]
        image = cv2.imread(image_path)

        if image is None:
            print("⚠️ Could not reload image for visualization")
            return

        h, w = image.shape[:2]

        colors = {
            0: (0, 0, 255),
            1: (0, 255, 255),
            2: (255, 0, 255),
            3: (0, 255, 0),
        }

        for det in result["detections"]:
            bbox = np.array(det["bbox"]).astype(int)
            cid = det["class_id"]
            conf = det["calibrated_confidence"]

            color = colors.get(cid, (255, 255, 255))

            cv2.rectangle(
                image,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                color,
                3,
            )

            cv2.putText(
                image,
                f"{cid}: {conf:.2f}",
                (bbox[0], bbox[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        banner_colors = {
            "SAFE": (0, 200, 0),
            "CAUTION": (0, 165, 255),
            "ABORT": (0, 0, 255),
        }

        decision = result["final_decision"]
        score = result["final_score"]

        cv2.rectangle(image, (0, 0), (w, 80), banner_colors[decision], -1)

        cv2.putText(
            image,
            f"{decision} | {score:.2f}",
            (20, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 255),
            4,
        )

        out_dir = Path("results/visualizations")
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"{Path(image_path).stem}_mc_dropout.jpg"

        cv2.imwrite(str(out_path), image)

        print(f"🖼️  Visualization saved → {out_path}")


# ---------------------------------------------------------
# DEMO
# ---------------------------------------------------------

if __name__ == "__main__":

    pipeline = MCDropoutPipeline()

    test_image = r"data\valid\valid\images\frame_000880.jpg"

    result = pipeline.process(test_image)

    pipeline.save_visualization(result)

    Path("results/metrics").mkdir(parents=True, exist_ok=True)

    json_path = "results/metrics/mc_dropout_demo.json"

    safe_result = make_json_safe(result)

    with open(json_path, "w") as f:
        json.dump(safe_result, f, indent=2)

    print("\n📄 Saved demo JSON →", json_path)
