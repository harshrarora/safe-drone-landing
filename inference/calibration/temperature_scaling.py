

"""
Temperature Scaling Calibration Module

Used in:
- Phase 4 calibration
- Phase 5 pipelines
- Evaluation scripts
"""

import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt


# ============================================================
# TEMPERATURE SCALING CLASS
# ============================================================

class TemperatureScaling:
    """Calibrate YOLO confidence scores using temperature scaling."""

    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.temperature = 1.0

    # --------------------------
    # Calibration
    # --------------------------

    def calibrate(
        self,
        val_dataset_path: str,
        save_path: str = "models/calibration/temperature.pkl",
        max_images: int = 500
    ):
        print("Collecting validation predictions...")

        logits_list = []
        labels_list = []

        val_images = (
            list(Path(val_dataset_path).glob("*.jpg")) +
            list(Path(val_dataset_path).glob("*.png"))
        )

        for img_path in tqdm(val_images[:max_images], desc="Processing"):
            try:
                results = self.model(str(img_path), verbose=False)

                label_path = (
                    str(img_path)
                    .replace("images", "labels")
                    .replace(img_path.suffix, ".txt")
                )

                if not Path(label_path).exists():
                    continue

                with open(label_path) as f:
                    for line in f:
                        gt_class = int(line.split()[0])
                        labels_list.append(gt_class)

                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        prob = float(box.conf[0])
                        logit = np.log(prob / (1 - prob + 1e-10))
                        logits_list.append(logit)

            except Exception:
                continue

        if not logits_list:
            raise RuntimeError("No validation predictions collected.")

        min_len = min(len(logits_list), len(labels_list))
        logits = torch.tensor(logits_list[:min_len], dtype=torch.float32)
        labels = torch.tensor(labels_list[:min_len], dtype=torch.float32)

        print("Optimizing temperature...")
        self.temperature = self._optimize_temperature(logits, labels)

        print(f"✅ Optimal temperature: {self.temperature:.3f}")

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump({"temperature": self.temperature}, f)

        print(f"Saved to: {save_path}")

    # --------------------------

    def _optimize_temperature(self, logits, labels):
        temperature = nn.Parameter(torch.ones(1) * 1.5)

        optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            scaled = logits / temperature
            loss = nn.functional.binary_cross_entropy_with_logits(
                scaled, labels
            )
            loss.backward()
            return loss

        optimizer.step(eval)
        return float(temperature.item())

    # --------------------------

    def apply_temperature(self, prob: float) -> float:
        return prob ** (1.0 / self.temperature)

    # --------------------------

    def predict_calibrated(self, image_path: str):

        results = self.model(image_path, verbose=False)
        detections = []

        if results[0].boxes is not None:
            for box in results[0].boxes:
                raw = float(box.conf[0])
                detections.append({
                    "class_id": int(box.cls[0]),
                    "raw_confidence": raw,
                    "calibrated_confidence": self.apply_temperature(raw),
                    "bbox": box.xyxy[0].cpu().numpy()
                })

        return {
            "detections": detections,
            "temperature": self.temperature
        }

    # --------------------------

    def load_temperature(self, path: str):
        with open(path, "rb") as f:
            self.temperature = pickle.load(f)["temperature"]

        print(f"✅ Loaded temperature: {self.temperature:.3f}")


# ============================================================
# CALIBRATION METRICS
# ============================================================

def compute_ece(confidences, accuracies, n_bins=10):

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)

        if mask.any():
            acc = accuracies[mask].mean()
            conf = confidences[mask].mean()
            ece += abs(conf - acc) * mask.mean()

    return ece


# ============================================================
# RELIABILITY DIAGRAM
# ============================================================

def plot_reliability_diagram(conf, acc, title, save_path):

    bins = np.linspace(0, 1, 11)
    xs, ys = [], []

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        xs.append(conf[mask].mean() if mask.any() else (lo + hi) / 2)
        ys.append(acc[mask].mean() if mask.any() else 0)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "--k")
    ax.bar(xs, ys, width=0.08)

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
