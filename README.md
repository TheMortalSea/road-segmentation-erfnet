Multiband Semantic Road Segmentation Using ERFNet
A lightweight deep learning approach for road segmentation in remote sensing imagery using ERFNet architecture with NAIP and LiDAR data fusion.
Overview
This project explores the effectiveness of ERFNet (Efficient Residual Factorized ConvNet) for semantic road segmentation in scenarios with limited training data. The study compares performance between standard 4-channel NAIP imagery and enhanced 6-channel datasets that incorporate LiDAR-derived features.
Key Features

Lightweight Architecture: Uses ERFNet for efficient real-time semantic segmentation
Multi-band Input: Supports both 4-channel NAIP and 6-channel NAIP+LiDAR configurations
Small Dataset Optimization: Designed to work effectively with limited training data
Geographic Focus: Tested on the complex terrain of Monterey Peninsula, California

Dataset
Input Channels

NAIP 4-band: Red, Green, Blue, Near-Infrared (NIR)
LiDAR-derived: Normalized Digital Surface Model (NDSM), Intensity
Total: Up to 6 input channels for enhanced spatial context

Study Area

Location: Monterey Peninsula, California
Rationale: Complex geography and varied terrain features provide robust testing conditions
Data Availability: Consistent LiDAR survey data available for the region

Methodology
Model Architecture

Base Model: ERFNet (Efficient Residual Factorized ConvNet)
Input: Variable channel input (4 or 6 channels)
Output: Binary road segmentation masks
Optimization: Dilated convolutions and residual connections for efficiency

Training Strategy

Cross-validation: 3-fold K-fold validation
Optimizer: Adam with learning rate 2e-4
Epochs: 10 per fold
Batch Size: 32
Loss Function: Combined Dice Loss and Binary Cross-Entropy with positive weights

Data Preprocessing

Filtering of background-only images
Pixel value normalization across all channels
Mask scaling to [0, 1] range
Data augmentation (flipping, rotation)

Results
Performance Comparison
ModelAccuracyPrecisionRecallF1 ScoreNAIP 4-band0.79650.33710.63960.4404NAIP + LiDAR 6-band0.78300.49490.86660.6040
Key Findings

Enhanced Precision: 47% improvement with LiDAR integration
Better Recall: 35% improvement in road pixel detection
Improved F1 Score: 37% increase in overall segmentation quality
Reduced False Positives: More accurate road boundary detection
