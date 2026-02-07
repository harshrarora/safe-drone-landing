# Safety-Critical Autonomous Drone Landing with Uncertainty Quantification

**HackTU 7.0**

## Problem

Autonomous drones must assess landing zone safety with calibrated confidence — not just detect landing pads. Over-confident predictions in defense scenarios can cause mission failure.

## Our Solution

We build a safety-aware perception stack that quantifies uncertainty before allowing a landing decision:

1. Monte-Carlo Dropout — stochastic inference to measure model instability  
2. Deep Ensembles — disagreement across independently trained models  
3. Multi-factor Risk Modeling — uncertainty + obstacles + image quality  
4. Calibrated Confidence — temperature scaling so probabilities reflect reality  

##  Current Results

### Phase 1: Baseline (Completed )
- Dataset: TEKNOFEST aerial drone footage (1920×1080, 4 classes)
- Model: YOLO11n
- Performance: mAP@0.5 = 0.953

### Phase 2: Data Augmentation (Completed ✅)
- Night simulation, blur, occlusion, weather, geometry
- Dataset expanded 4×
- Augmented model mAP@0.5 = 0.918

### Phase 3: Uncertainty Quantification (In Progress)

#### Monte-Carlo Dropout (Completed )
- Backbone-level dropout via forward hooks
- 30 stochastic passes
- Metrics: consensus, bbox variance, confidence spread

#### Deep Ensembles (Running )
- 5 models trained with varied seeds & augmentation
- Currently training remaining members

### Phase 4: Calibration (Running )
- Temperature scaling
- Reliability curves
- Expected Calibration Error


