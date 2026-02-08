"""
Deep Ensemble + Monte Carlo Dropout uncertainty module
"""

from pathlib import Path
import numpy as np

from inference.uncertainty.mc_dropout import MCDropoutYOLO


class EnsembleMCDropout:
    """
    Runs MC Dropout on each ensemble member and aggregates.
    """

    def __init__(
        self,
        ensemble_root: str,
        n_iterations: int = 20,
        dropout_rate: float = 0.2,
    ):

        self.models = []

        ensemble_root = Path(ensemble_root)

        members = sorted([p for p in ensemble_root.iterdir() if p.is_dir()])

        if not members:
            raise RuntimeError(f"No ensemble members found in {ensemble_root}")

        print("\n🧠 Initializing Ensemble MC Dropout:")
        print("-" * 50)

        for m in members:
            weights = m / "best.pt"

            if not weights.exists():
                raise RuntimeError(f"Missing best.pt in {m}")

            print(f"  → Loading {weights}")

            model = MCDropoutYOLO(
                model_path=str(weights),
                n_iterations=n_iterations,
                dropout_rate=dropout_rate,
            )

            self.models.append(model)

        print(f"✅ Loaded {len(self.models)} ensemble members")

    # --------------------------------------------------

    def predict(self, image_path: str):

        all_consensus = []
        all_bbox_uncertainty = []
        first_detections = None

        print("\n🔁 Running Ensemble + MC Dropout")

        for idx, model in enumerate(self.models, 1):

            print(f"\n--- Member {idx}/{len(self.models)} ---")

            result = model.predict_with_uncertainty(image_path)

            all_consensus.append(float(result["consensus"]))
            all_bbox_uncertainty.append(float(result["bbox_uncertainty"]))

            if first_detections is None:
                first_detections = result["detections"]

        return {
            "consensus": float(np.mean(all_consensus)),
            "bbox_uncertainty": float(np.mean(all_bbox_uncertainty)),
            "detections": first_detections,
            "num_models": len(self.models),
        }
