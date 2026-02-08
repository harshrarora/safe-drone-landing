# Safety-Critical Autonomous Drone Landing with Uncertainty Quantification  
**HackTU 7.0**

Autonomous drones must assess landing zones with **calibrated confidence**, not blind certainty.  
In defense and emergency scenarios, over-confident perception systems can cause catastrophic failures.

This project builds a **risk-aware landing decision engine** combining deep learning detection, Bayesian uncertainty estimation, calibration, and multi-factor safety logic.

## Key Features

- **Monte-Carlo Dropout** — stochastic inference to measure prediction instability  
- **Deep Ensembles** — disagreement across independently trained models  
- **Confidence Calibration** — temperature scaling for honest probabilities  
- **Multi-Factor Risk Engine** — obstacles, image quality, uncertainty, terrain  
- **Decision System** — SAFE / CAUTION / ABORT with explanations  
- **Visual Overlays** — bounding boxes + safety banners + uncertainty stats

---

##  Current Progress

### Phase 1 — Baseline Detection
- Dataset: TEKNOFEST aerial footage (1920×1080, 4 classes)
- Model: YOLO11n
- mAP@0.5: **0.953**

### Phase 2 — Robust Training via Augmentation
- Night simulation, blur, fog, occlusion, weather
- Dataset expanded **4×**
- Augmented model mAP@0.5: **0.918**

### Phase 3 — Uncertainty Quantification 

#### Monte-Carlo Dropout
- Backbone-level dropout via forward hooks
- 30 stochastic passes
- Metrics: consensus, bbox variance, confidence spread

#### Deep Ensembles
- 3 models with varied seeds & training schedules  
- Remaining members currently training

### Phase 4 — Calibration
- Temperature scaling
- Reliability curves
- Expected Calibration Error (ECE)

### Phase 5 — Risk-Aware Landing Logic
- Altitude-aware obstacle clearance
- Blur & lighting detection
- Terrain complexity analysis
- Edge proximity checks
- Additive penalty scoring with audit logging

## System Architecture

Camera Feed
↓
YOLO Detection
↓
MC Dropout + Ensemble
↓
Calibration
↓
Risk Assessment Engine
↓
SAFE / CAUTION / ABORT

## What Makes This Different
Most teams only report detection accuracy.

We explicitly answer: When should the drone NOT trust itself?

| Feature | Standard YOLO | Our System |
|--------|-------------|------------|
| Calibrated Confidence | ❌ | ✅ |
| Uncertainty Quantification | ❌ | ✅ |
| Risk Reasoning | ❌ | ✅ |
| Safety-aware Decision | ❌ | ✅ |


## HackTU 7.0 Submission
This project focuses on responsible deployment of autonomous perception systems, targeting real-world safety-critical UAV operations.
