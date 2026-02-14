# Music Genre Classification from Noisy Audio

This project implements deep learning models for classifying music genres from noisy audio mixtures. Built for the BSDA2001P Deep Learning and GenAI course Kaggle competition.

## Problem Statement

The task is to classify songs into 10 genres (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock) using audio that has been degraded with noise and tempo variations. Each training song is provided as four separate stems (bass, drums, other, vocals), while test data consists of noisy mashups created by mixing these stems.

## Dataset

- Training: 1000 songs, 100 per genre
- Each song has 4 audio stems (bass.wav, drums.wav, other.wav, vocals.wav)
- Test: 3020 noisy mashup files with tempo variations and environmental noise
- Audio format: WAV files, approximately 30 seconds each
- Sample rate: 22050 Hz (resampled from 44100 Hz)

## Approach

### Feature Extraction
- Mel-spectrograms with 128 mel bands
- Fixed shape: 128 x 1292 (time-frequency representation)
- Power to decibel conversion for improved dynamic range

### Data Augmentation
- Cross-song stem mixing with random weights
- Time stretching (0.85x to 1.15x)
- Pitch shifting (±2 semitones)
- ESC-50 environmental noise injection (SNR 10-30 dB)
- Gaussian noise addition
- Augmentation factor: 5x (800 original songs → 4800 training samples)

### Models Implemented

1. **Custom CNN**
   - 4 convolutional blocks with batch normalization
   - Global average pooling and fully connected layers
   - Parameters: ~1.2M
   - Validation F1: 0.6931

2. **Pretrained ResNet18** (Best Model)
   - Transfer learning from ImageNet
   - Modified first conv layer for single-channel input
   - Fine-tuned all layers
   - Parameters: ~11M
   - Validation F1: 0.8751

3. **Depthwise Separable CNN**
   - MobileNet-style architecture
   - Efficient depthwise separable convolutions
   - Parameters: ~3.5M
   - Validation F1: 0.8216

### Ensemble
- Weighted voting based on validation F1 scores
- Combines predictions from all three models

## Results

| Model | Validation F1 | Parameters | Training Time/Epoch |
|-------|---------------|------------|---------------------|
| Custom CNN | 0.6931 | 1.2M | ~40s |
| ResNet18 | 0.8751 | 11M | ~70s |
| Depthwise CNN | 0.8216 | 3.5M | ~45s |
| Ensemble | - | - | - |

The ResNet18 model exceeded the target F1 score of 0.80 and was selected for final submission.

## Installation

Install required packages:

```bash
pip install -r requirements.txt
```

## Usage

The main notebook contains the complete pipeline:

1. Load and preprocess audio data
2. Extract mel-spectrogram features
3. Apply data augmentation
4. Train three different models
5. Evaluate and compare performance
6. Generate predictions for test set

All experiments are logged to Weights and Biases for tracking.

## Project Structure

```
.
├── requirements.txt          # Python dependencies
├── README.md                # Project documentation
└── notebook.ipynb           # Main implementation notebook
```

## Key Learnings

- Transfer learning from ImageNet works well for audio spectrograms
- Aggressive data augmentation is critical for handling distribution shift
- Depthwise separable convolutions offer good efficiency-performance trade-off
- Proper experiment tracking with wandb helps compare architectures

## Future Improvements

- Explore larger models like ResNet50 or EfficientNet
- Implement attention mechanisms for temporal feature selection
- Use longer audio segments to capture more musical context
- Try alternative representations like constant-Q transforms
- Systematic hyperparameter optimization

## References

- He et al. (2016) - Deep Residual Learning for Image Recognition
- Howard et al. (2017) - MobileNets: Efficient CNNs for Mobile Vision
- McFee et al. (2015) - librosa: Audio and Music Signal Analysis
- Piczak (2015) - ESC Dataset for Environmental Sound Classification

## Author

Student ID: 24f2001946-iitm  
Course: BSDA2001P Introduction to Deep Learning and GenAI  
Competition: January 2026 DL GenAI Project
