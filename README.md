# Ocean Sound Classification using Deep Learning

A comprehensive deep learning project for classifying marine mammal sounds using multiple neural network architectures. This project compares the performance of DNNs, CNNs, RNNs, LSTMs, Transformers, and raw waveform models on ocean audio data.

## Project Overview

This project explores various deep learning approaches for classifying marine mammal sounds from the **Watkins Marine Mammal Sound (WMMS)** dataset. The goal is to understand how different neural network architectures perform on audio classification tasks and compare spectrogram-based methods with raw waveform analysis.

## Objectives

- Build and evaluate multiple deep learning models for ocean sound classification
- Compare different neural network families (DNN, CNN, RNN, LSTM, Transformer)
- Analyze the effectiveness of spectrogram representations vs. raw audio waveforms
- Understand audio preprocessing and feature extraction techniques

## Dataset

The project uses the **Watkins Marine Mammal Sound (WMMS)** dataset, which includes:
- WAV audio clips of marine mammal sounds (whales, dolphins, etc.)
- CSV file with labels for each audio sample
- Multiple species classifications

## Models Implemented

### 1. **Spectrogram + Deep Neural Network (DNN)**
- Converts audio clips into spectrograms
- Flattens 2D spectrograms into 1D feature vectors
- Multi-layer dense neural network
- **Accuracy: 54.41%**

### 2. **Spectrogram + Convolutional Neural Network (CNN)**
- Treats spectrograms as 2D images
- Utilizes convolutional layers for spatial pattern recognition
- Best performing model
- **Accuracy: 75.00%** (Best performer)

### 3. **Spectrogram + Recurrent Neural Network (RNN)**
- Processes spectrograms as sequential data over time
- Uses recurrent layers for temporal pattern learning
- **Accuracy: 21.57%**

### 4. **Spectrogram + Long Short-Term Memory (LSTM)**
- Enhanced RNN with memory cells
- Better at capturing long-term dependencies
- **Accuracy: 28.92%**

### 5. **Spectrogram + Transformer**
- Uses attention mechanisms for sequence processing
- Self-attention across time steps
- **Accuracy: 6.86%**

### 6. **Raw Waveform Classification**
- Direct classification from audio waveforms (no spectrograms)
- Tests the necessity of feature engineering
- **Accuracy: 2.94%**

## Results Summary

| Model | Accuracy | Notes |
|-------|----------|-------|
| **CNN** | **75.00%** | Best performer - leverages 2D spatial features |
| DNN | 54.41% | Good performance with flattened features |
| LSTM | 28.92% | Better than RNN but still limited |
| RNN | 21.57% | Struggles with 2D time-frequency data |
| Transformer | 6.86% | Attention alone insufficient for this task |
| Raw Waveform | 2.94% | Demonstrates need for spectrograms |

## Key Findings

**Why CNN Won:**
- Spectrograms behave like images with both time and frequency dimensions
- CNNs excel at detecting spatial patterns in 2D data
- Convolutional layers effectively capture frequency patterns and temporal changes
- Spatial hierarchies match the nature of audio spectrograms

**Why Sequential Models Struggled:**
- RNNs and LSTMs treat data as 1D sequences
- They lose the crucial 2D time-frequency structure
- Cannot effectively model both dimensions simultaneously

**Why Raw Waveforms Failed:**
- Raw audio is too high-dimensional and complex
- Lacks the informative time-frequency representation
- Spectrograms provide essential feature extraction

## Technologies Used

- **Python 3.x**
- **TensorFlow / Keras** - Deep learning framework
- **Librosa** - Audio processing and feature extraction
- **NumPy & Pandas** - Data manipulation
- **Matplotlib & Seaborn** - Visualization
- **Scikit-learn** - Model evaluation and metrics
- **Google Colab** - GPU acceleration

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ocean-sound-classification.git
cd ocean-sound-classification

# Install required packages
pip install librosa matplotlib numpy pandas scikit-learn tensorflow tqdm seaborn
```

## Usage

1. **Prepare Dataset:**
   - Upload your CSV labels file
   - Upload audio archive (ZIP or RAR format)

2. **Run the Notebook:**
   ```python
   # Open in Google Colab or Jupyter Notebook
   jupyter notebook AhmetAytac_Assigment5.ipynb
   ```

3. **Follow the workflow:**
   - Data loading and preprocessing
   - Spectrogram generation
   - Model training for each architecture
   - Performance evaluation and comparison

## Model Architecture Details

### CNN Architecture
```
Conv2D → BatchNorm → MaxPooling → Dropout
Conv2D → BatchNorm → MaxPooling → Dropout
Flatten → Dense → Dropout → Dense (Softmax)
```

### DNN Architecture
```
Flatten → Dense → Dropout
Dense → Dropout
Dense → Dropout
Dense (Softmax)
```

### LSTM Architecture
```
LSTM → Dropout
LSTM → Dropout
Dense (Softmax)
```

## Evaluation Metrics

- **Accuracy Score**
- **Confusion Matrix**
- **Classification Report** (Precision, Recall, F1-Score)
- **Training/Validation Loss Curves**
- **Training/Validation Accuracy Curves**

## Course Information

**Course:** CAP4770 - Deep Learning  
**Assignment:** Assignment 5 - Ocean Sound Classification  

**Date:** November 26, 2025  

## Future Improvements

- [ ] Data augmentation techniques (time stretching, pitch shifting)
- [ ] Ensemble methods combining multiple models
- [ ] Transfer learning from pre-trained audio models
- [ ] Attention mechanisms specifically designed for audio
- [ ] Mel-frequency cepstral coefficients (MFCCs) as additional features
- [ ] Hybrid CNN-RNN architectures

## License

This project is created for educational purposes as part of a university course assignment.

## Contributing

Feel free to fork this repository and submit pull requests for any improvements!

## Contact

For questions or collaborations, please reach out via GitHub issues.

---

**Star this repo if you found it helpful!**
