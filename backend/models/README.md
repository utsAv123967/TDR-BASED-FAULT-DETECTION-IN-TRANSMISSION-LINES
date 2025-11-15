# Model Files

This folder should contain your trained PyTorch model:

- `image_fault_classifier.pth` - ResNet18 model for fault classification (Open/Short/Resistive)

## How to get the model:

1. Download `image_fault_classifier.pth` from your Google Drive (where you trained it in Colab)
2. Place it in this `models/` folder
3. The backend will automatically load it on startup

## Model Details:

- Architecture: ResNet18
- Input: 224x224 RGB image of TDR waveform
- Output: 3 classes
  - Class 0: Open → "No fault"
  - Class 1: Short → "Short Circuit"  
  - Class 2: Resistive → "Resistive Fault"
