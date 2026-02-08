"""
FINAL PIPELINE — Deep Ensemble + MC Dropout + Calibration + Risk
"""

import sys
from pathlib import Path
import json
import numpy as np
import cv2

# ---------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from inference.uncertainty.ensemble_mc_dropout import EnsembleMCDropout
from inference.calibration.temperature_scaling import TemperatureScaling
from inference.risk_assessment.safety_evaluator import SafetyEvaluator


# ---------------------------------------------------------
# JSON UTILITY
# ---------------------------------------------------------

def make_json_safe(obj):
    import numpy as np

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

class EnsembleMCPipeline:

    def __init__(
        self,
        ensemble_root="models/yolo/ensemble",
        temperature_path="models/calibration/temperature_augmented.pkl",
    ):

        print("\n🚀 Initializing ENSEMBLE + MC Pipeline...")

        self.uncertainty = EnsembleMCDropout(
            ensemble_root=ensemble_root,
            n_iterations=20,
        )

        # Calibration uses first model
        first_model = sorted(Path(ensemble_root).glob("*"))[0] / "best.pt"

        self.calibrator = TemperatureScaling(str(first_model))
        self.calibrator.load_temperature(temperature_path)

        self.safety = SafetyEvaluator()

        print("✅ Ensemble pipeline ready!")

    # --------------------------------------------------

    def process(self, image_path: str):

        print("\n" + "=" * 70)
        print(f"🛰️  Processing: {image_path}")

        # --------------------------------------------------
        # 1. Ensemble + MC Dropout
        # --------------------------------------------------

        result = self.uncertainty.predict(image_path)

        consensus = result["consensus"]
        uncertainty = result["bbox_uncertainty"]

        print(f"→ Ensemble consensus: {consensus:.2%}")
        print(f"→ Ensemble bbox uncertainty: {uncertainty:.4f}")

        # --------------------------------------------------
        # 2. Calibration
        # --------------------------------------------------

        cal = self.calibrator.predict_calibrated(image_path)
        detections = cal["detections"]

        # --------------------------------------------------
        # 3. Safety (SAFE WRAPPER)
        # --------------------------------------------------

        try:
            safety = self.safety.assess_landing_safety(
                image_path=image_path,
                detections=detections,
                uncertainty=uncertainty,
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

        return {
            "image": image_path,
            "mode": "ensemble_mc",
            "uncertainty": {
                "consensus": consensus,
                "bbox_uncertainty": uncertainty,
                "num_models": result["num_models"],
            },
            "detections": detections,
            "safety": safety,
            "decision": safety["decision"],
            "score": safety["safety_score"],
        }

    # --------------------------------------------------

    def save_visualization(self, result: dict):

        img = cv2.imread(result["image"])

        if img is None:
            print("⚠️ Could not reload image for visualization")
            return

        h, w = img.shape[:2]

        # Draw detections
        colors = {
            0: (0, 0, 255),
            1: (0, 255, 255),
            2: (255, 0, 255),
            3: (0, 255, 0),
        }

        for det in result["detections"]:
            box = np.array(det["bbox"]).astype(int)
            cls = det["class_id"]
            conf = det["calibrated_confidence"]

            color = colors.get(cls, (255, 255, 255))

            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 3)
            cv2.putText(
                img,
                f"{cls}:{conf:.2f}",
                (box[0], box[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        banner = {
            "SAFE": (0, 200, 0),
            "CAUTION": (0, 165, 255),
            "ABORT": (0, 0, 255),
        }[result["decision"]]

        cv2.rectangle(img, (0, 0), (w, 90), banner, -1)

        cv2.putText(
            img,
            f"{result['decision']} | {result['score']:.2f}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 255),
            4,
        )

        out_dir = Path("results/visualizations")
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"{Path(result['image']).stem}_ensemble_mc.jpg"

        cv2.imwrite(str(out_path), img)

        print(f"🖼️ Saved visualization → {out_path}")


# --------------------------------------------------
# DEMO
# --------------------------------------------------

if __name__ == "__main__":

    pipeline = EnsembleMCPipeline()

    img = r"data\valid\valid\images\frame_000880.jpg"

    res = pipeline.process(img)

    Path("results/metrics").mkdir(parents=True, exist_ok=True)

    safe = make_json_safe(res)

    with open("results/metrics/ensemble_mc_demo.json", "w") as f:
        json.dump(safe, f, indent=2)

    pipeline.save_visualization(res)

    print("\n🏁 FINAL DECISION:", res["decision"])
