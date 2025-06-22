# Multiband Semantic Road Segmentation Using ERFNet

A lightweight deep learning approach for road segmentation in remote sensing imagery using ERFNet architecture with NAIP and LiDAR data fusion.

## Overview

This project explores the effectiveness of ERFNet (Efficient Residual Factorized ConvNet) for semantic road segmentation in scenarios with limited training data. The study compares performance between standard 4-channel NAIP imagery and enhanced 6-channel datasets that incorporate LiDAR-derived features.

## Key Features

- **Lightweight Architecture**: Uses ERFNet for efficient real-time semantic segmentation
- **Multi-band Input**: Supports both 4-channel NAIP and 6-channel NAIP+LiDAR configurations
- **Small Dataset Optimization**: Designed to work effectively with limited training data
- **Geographic Focus**: Tested on the complex terrain of Monterey Peninsula, California

## Dataset

### Input Channels
- **NAIP 4-band**: Red, Green, Blue, Near-Infrared (NIR)
- **LiDAR-derived**: Normalized Digital Surface Model (NDSM), Intensity
- **Total**: Up to 6 input channels for enhanced spatial context

### Study Area
- **Location**: Monterey Peninsula, California
- **Rationale**: Complex geography and varied terrain features provide robust testing conditions
- **Data Availability**: Consistent LiDAR survey data available for the region

## Methodology

### Model Architecture
- **Base Model**: ERFNet (Efficient Residual Factorized ConvNet)
- **Input**: Variable channel input (4 or 6 channels)
- **Output**: Binary road segmentation masks
- **Optimization**: Dilated convolutions and residual connections for efficiency

### Training Strategy
- **Cross-validation**: 3-fold K-fold validation
- **Optimizer**: Adam with learning rate 2e-4
- **Epochs**: 10 per fold
- **Batch Size**: 32
- **Loss Function**: Combined Dice Loss and Binary Cross-Entropy with positive weights

### Data Preprocessing
- Filtering of background-only images
- Pixel value normalization across all channels
- Mask scaling to [0, 1] range
- Data augmentation (flipping, rotation)

## Results

### Performance Comparison

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| NAIP 4-band | 0.7965 | 0.3371 | 0.6396 | 0.4404 |
| NAIP + LiDAR 6-band | 0.7830 | 0.4949 | 0.8666 | 0.6040 |

### Key Findings
- **Enhanced Precision**: 47% improvement with LiDAR integration
- **Better Recall**: 35% improvement in road pixel detection
- **Improved F1 Score**: 37% increase in overall segmentation quality
- **Reduced False Positives**: More accurate road boundary detection

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/multiband-road-segmentation.git
cd multiband-road-segmentation

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training
```python
# Train with 4-band NAIP data
python train.py --channels 4 --data_path /path/to/naip/data

# Train with 6-band NAIP+LiDAR data
python train.py --channels 6 --data_path /path/to/multiband/data
```

### Inference
```python
# Run inference on new imagery
python predict.py --model_path /path/to/trained/model --input_image /path/to/image
```

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- numpy
- opencv-python
- scikit-learn
- matplotlib

## File Structure

```
├── README.md
├── requirements.txt
├── train.py
├── predict.py
├── models/
│   └── erfnet.py
├── data/
│   ├── preprocessing.py
│   └── augmentation.py
├── utils/
│   ├── loss_functions.py
│   └── metrics.py
└── notebooks/
    └── analysis.ipynb
```

## Applications

- **Urban Planning**: Infrastructure monitoring and development
- **Disaster Response**: Rapid road condition assessment
- **Autonomous Vehicles**: Real-time road detection for navigation
- **Remote Sensing**: Large-scale transportation network mapping

## Future Work

- **Dataset Expansion**: Include more diverse regions and road types
- **Architecture Exploration**: Test hybrid models and alternative lightweight architectures
- **Transfer Learning**: Leverage pre-trained models for improved small dataset performance
- **Advanced Augmentation**: Implement sophisticated data augmentation techniques

## Citation

If you use this work in your research, please cite:

```bibtex
@article{road_segmentation_2024,
  title={Multiband Semantic Road Segmentation Comparison Using ERFNet Architecture in Monterey, CA},
  author={[Your Name]},
  journal={UCL Department of Geography},
  year={2024},
  note={Advanced Topics in Social and Geographic Data Science}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- UCL Department of Geography
- National Agriculture Imagery Program (NAIP)
- ERFNet architecture by Romera et al. (2018)
- Claude.ai for development assistance

## Contact

For questions or collaboration opportunities, please open an issue or contact [your.email@domain.com].