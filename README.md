# Intelligent Video Surveillance System

## Overview
An intelligent video surveillance system that uses deep learning for anomaly detection and object recognition.

## Prerequisites
- Python 3.8+
- TensorFlow 2.x
- OpenCV
- Other dependencies listed in `requirements.txt`

## Setup

1. **Create virtual environment:**
   ```bash
   python -m venv myenv
   myenv\Scripts\activate  # Windows
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the autoencoder model (Required before first run):**
   ```bash
   python train_model.py
   ```
   This will create the `models/autoencoder.h5` file needed by the application.

## Usage

```bash
python app.py
```

## Project Structure
```
Intelligent-Video-Surveillance-System/
├── models/
│   └── autoencoder.h5  # Trained model (created after training)
├── app.py              # Main application
├── train_model.py      # Model training script
└── requirements.txt    # Python dependencies
```

## Troubleshooting

### OSError: Unable to synchronously open file
This error occurs when the model file doesn't exist. Solution:
1. Run `python train_model.py` to train and save the model first
2. Ensure the `models/` directory exists
3. Check that `models/autoencoder.h5` was created successfully

## Notes
- The system uses TensorFlow with oneDNN optimizations enabled by default
- First run may take longer due to TensorFlow initialization
- Ensure you have sufficient disk space for model files
