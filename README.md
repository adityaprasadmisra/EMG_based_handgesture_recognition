# EMG-Based Hand Gesture Recognition Using ESP32 and Machine Learning

## Overview

This project presents an **EMG (Electromyography)-Based Hand Gesture Recognition System** that uses forearm muscle signals to classify hand gestures. The system combines **EMG signal acquisition, signal processing, feature extraction, and machine learning** to recognize different gestures in real time.

The project demonstrates how biological signals can be transformed into meaningful commands for applications such as prosthetic control, human-computer interaction, rehabilitation systems, and wearable technology.

---

## Features

* Real-time EMG signal acquisition using ESP32
* Dual-channel EMG data collection
* Signal quality analysis (PSD and SNR)
* Noise reduction using digital filtering
* Statistical feature extraction
* Machine Learning based gesture classification
* Multiple model comparison (KNN, SVM, Random Forest)
* Dataset collection from multiple users
* Model persistence using Pickle

---

## Hand Gestures Recognized

The system classifies the following five gestures:

1. **Relax**
2. **Fist**
3. **Flexor**
4. **Extensor**
5. **Open Hand**

---

## Hardware Components

| Component            | Purpose                            |
| -------------------- | ---------------------------------- |
| ESP32                | Data acquisition and communication |
| EMG Muscle Sensor V3 | Muscle signal amplification        |
| Surface Electrodes   | EMG signal pickup                  |
| USB Serial Interface | Data transfer to PC                |
| MPU6050 (Optional)   | Motion sensing and sensor fusion   |

---

## System Architecture

```text
Forearm Muscles
        │
        ▼
EMG Sensors
        │
        ▼
ESP32 ADC
        │
        ▼
Serial Communication
        │
        ▼
Python Data Collection
        │
        ▼
Signal Processing
        │
        ▼
Feature Extraction
        │
        ▼
Machine Learning Model
        │
        ▼
Gesture Prediction
```

---

## Dataset Collection

Data was collected using a custom Python-based acquisition tool connected to the ESP32 through serial communication.

For each gesture:

1. User prepares the gesture.
2. Settling time is provided.
3. EMG signals are recorded.
4. Multiple trials are performed.
5. Data is stored in CSV format.

Example dataset format:

```csv
emg1,emg2,gesture
2345,1780,fist
2410,1805,fist
1720,1500,relax
```

---

## Signal Analysis

Before training, signal quality was evaluated using:

### Baseline Noise Analysis

The relax gesture was used to estimate:

* Mean signal level
* Standard deviation
* Noise floor

### Power Spectral Density (PSD)

Welch's Method was used to analyze frequency content and compare active versus inactive muscle states.

### Signal-to-Noise Ratio (SNR)

SNR was calculated to quantify the quality of acquired EMG signals.

---

## Signal Processing Pipeline

```text
Raw EMG Signal
       │
       ▼
50 Hz Notch Filter
       │
       ▼
20–450 Hz Bandpass Filter
       │
       ▼
Clean EMG Signal
```

### Notch Filter

Removes power-line interference at 50 Hz.

### Bandpass Filter

Preserves useful EMG frequencies while removing motion artifacts and high-frequency noise.

---

## Feature Extraction

Raw EMG signals are segmented using sliding windows and converted into statistical features.

Extracted features:

### Mean Absolute Value (MAV)

Measures average muscle activation.

### Root Mean Square (RMS)

Represents signal energy.

### Variance

Measures signal spread.

### Standard Deviation

Measures fluctuations around the mean.

### Zero Crossing Count

Captures frequency-related characteristics.

---

## Machine Learning Pipeline

```text
Dataset
    │
    ▼
Data Cleaning
    │
    ▼
Feature Extraction
    │
    ▼
Label Encoding
    │
    ▼
Train-Test Split
    │
    ▼
Feature Scaling
    │
    ▼
Model Training
    │
    ▼
Evaluation
    │
    ▼
Best Model Selection
```

---

## Models Evaluated

### K-Nearest Neighbors (KNN)

Distance-based classification algorithm used as a baseline model.

### Support Vector Machine (SVM)

Implemented using the RBF kernel to classify nonlinear gesture patterns.

### Random Forest

Ensemble-based classifier consisting of multiple decision trees.

Random Forest achieved the best overall performance due to its robustness against noisy EMG data.

---

## Evaluation Metrics

Models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Cross Validation

---

## Results

* Successfully classified five hand gestures.
* Signal quality improved significantly after filtering.
* Random Forest demonstrated the best overall classification performance.
* Cross-validation confirmed model consistency across different data splits.

---
---

## Technologies Used

### Programming

* Python
* Embedded C

### Libraries

* NumPy
* Pandas
* Matplotlib
* SciPy
* Scikit-Learn
* Pickle

### Hardware

* ESP32
* EMG Muscle Sensor V3
* Surface Electrodes
* MPU6050

---

## Future Improvements

* Increase the number of supported gestures
* Collect larger multi-user datasets
* Explore deep learning models such as CNNs and LSTMs
* Implement wireless real-time prediction
* Develop prosthetic and rehabilitation applications

---

## Applications

* Prosthetic Hand Control
* Gesture-Controlled Robotics
* Human-Computer Interaction
* Rehabilitation Systems
* Wearable Healthcare Devices
* Assistive Technologies

---

## Skills Gained

* Embedded Systems
* Sensor Interfacing
* Serial Communication
* Signal Processing
* Machine Learning
* Feature Engineering
* Data Analysis
* Model Evaluation

---

## Conclusion

This project demonstrates the successful integration of embedded systems, biomedical signal processing, and machine learning for hand gesture recognition. By combining EMG signal acquisition, preprocessing, feature extraction, and classification techniques, a complete end-to-end gesture recognition pipeline was developed. The project provides a strong foundation for future work in wearable computing, assistive technology, and intelligent human-machine interfaces.
