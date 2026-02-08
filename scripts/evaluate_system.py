"""
Comprehensive evaluation of MC Dropout landing safety pipeline.

Runs the system on many images and generates:
- Decision distribution
- Safety score stats
- Uncertainty stats
- FPS
- Plots for submission
"""

from pathlib import Path
import json
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from inference.pipeline_mc_dropout import MCDropoutPipeline


# -----------------------------------------------------------
# DATASET EVALUATION
# -----------------------------------------------------------

def evaluate_on_dataset(pipeline, image_folder, max_images=50):

    images = list(Path(image_folder).glob("*.jpg")) + \
             list(Path(image_folder).glob("*.png"))

    images = images[:max_images]

    results = {
        "decisions": {"SAFE": 0, "CAUTION": 0, "ABORT": 0},
        "safety_scores": [],
        "bbox_uncertainties": [],
        "consensus_scores": [],
        "processing_times": [],
        "risk_counts": []
    }

    print(f"\n📂 Evaluating {len(images)} images...\n")

    for img_path in tqdm(images):

        start = time.time()

        try:
            output = pipeline.process(str(img_path))
        except Exception as e:
            print(f"❌ Failed on {img_path.name}: {e}")
            continue

        elapsed = time.time() - start

        results["decisions"][output["final_decision"]] += 1
        results["safety_scores"].append(output["final_score"])
        results["bbox_uncertainties"].append(
            output["uncertainty"]["bbox_uncertainty"]
        )
        results["consensus_scores"].append(
            output["uncertainty"]["consensus"]
        )
        results["processing_times"].append(elapsed)
        results["risk_counts"].append(
            len(output["safety_assessment"]["risk_factors"])
        )

    return results


# -----------------------------------------------------------
# REPORT + SAVE
# -----------------------------------------------------------

def summarize_results(results):

    print("\n" + "="*60)
    print("📊 SYSTEM EVALUATION SUMMARY")
    print("="*60)

    total = sum(results["decisions"].values())

    print("\nDecision Distribution:")
    for k, v in results["decisions"].items():
        pct = 100 * v / max(total, 1)
        print(f"  {k}: {v} ({pct:.1f}%)")

    print("\nSafety Scores:")
    print(f"  Mean: {np.mean(results['safety_scores']):.3f}")
    print(f"  Std:  {np.std(results['safety_scores']):.3f}")

    print("\nUncertainty:")
    print(f"  Mean bbox uncertainty: {np.mean(results['bbox_uncertainties']):.3f}")
    print(f"  Mean consensus: {np.mean(results['consensus_scores']):.2%}")

    avg_time = np.mean(results["processing_times"])

    print("\nPerformance:")
    print(f"  Avg time/image: {avg_time:.2f}s")
    print(f"  FPS: {1/avg_time:.2f}")

    return {
        "decision_distribution": results["decisions"],
        "safety_score_mean": float(np.mean(results["safety_scores"])),
        "uncertainty_mean": float(np.mean(results["bbox_uncertainties"])),
        "consensus_mean": float(np.mean(results["consensus_scores"])),
        "avg_time": float(avg_time),
        "fps": float(1/avg_time)
    }


def create_plots(results):

    Path("results/plots").mkdir(parents=True, exist_ok=True)

    # --- Safety score histogram ---
    plt.figure()
    plt.hist(results["safety_scores"], bins=20)
    plt.title("Safety Score Distribution")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.savefig("results/plots/safety_scores.png")

    # --- Uncertainty histogram ---
    plt.figure()
    plt.hist(results["bbox_uncertainties"], bins=20)
    plt.title("BBox Uncertainty")
    plt.xlabel("Uncertainty")
    plt.ylabel("Count")
    plt.savefig("results/plots/bbox_uncertainty.png")

    # --- Decision bar chart ---
    plt.figure()
    plt.bar(
        results["decisions"].keys(),
        results["decisions"].values()
    )
    plt.title("Decision Distribution")
    plt.savefig("results/plots/decision_distribution.png")

    print("\n📈 Plots saved to results/plots/")


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------

if __name__ == "__main__":

    pipeline = MCDropoutPipeline()

    IMAGE_FOLDER = "data/valid/valid/images"

    results = evaluate_on_dataset(
        pipeline,
        IMAGE_FOLDER,
        max_images=30
    )

    summary = summarize_results(results)

    Path("results/metrics").mkdir(parents=True, exist_ok=True)

    with open("results/metrics/system_eval.json", "w") as f:
        json.dump(summary, f, indent=2)

    create_plots(results)

    print("\n✅ Evaluation complete!")
