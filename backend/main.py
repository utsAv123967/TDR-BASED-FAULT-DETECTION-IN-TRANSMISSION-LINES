# TDR Cable Fault Classification - Simplified Backend
# This backend only performs fault classification using ResNet18

import os
import sys

# Fix Windows DLL loading issues
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO, StringIO
import base64

# Preprocessing matching training code
try:
    from scipy.signal import savgol_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not installed, using simple smoothing")

def simple_smooth(data, window_size=5):
    """Simple moving average smoothing (fallback)"""
    if len(data) < window_size:
        return data
    smoothed = np.copy(data).astype(float)
    for i in range(len(data)):
        start = max(0, i - window_size // 2)
        end = min(len(data), i + window_size // 2 + 1)
        smoothed[i] = np.mean(data[start:end])
    return smoothed

def preprocess_voltage(voltage):
    """Preprocess voltage data matching training code"""
    # Apply Savitzky-Golay filter if available (same as training)
    if HAS_SCIPY and len(voltage) > 50:
        v = savgol_filter(voltage, 51, 3)
    else:
        v = voltage.copy()
    
    # Normalize: center and scale (same as training)
    v = v - np.mean(v)
    vmax = np.max(np.abs(v)) + 1e-9
    v = v / vmax
    
    return v

# PyTorch imports - lazy load to avoid hangs
import torch
import torch.nn as nn
transforms = None
resnet18 = None

def lazy_import_torchvision():
    global transforms, resnet18
    if transforms is None:
        from torchvision import transforms as tv_transforms
        from torchvision.models import resnet18 as tv_resnet18
        transforms = tv_transforms
        resnet18 = tv_resnet18

app = FastAPI(title="TDR Fault Classification API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
device = "cuda" if torch.cuda.is_available() else "cpu"
classifier_model = None
distance_regressor = None
meta_ct = None

# Class names for classification
CLASS_NAMES = ['No Fault (Open)', 'Short Circuit', 'Resistive Fault']

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def clean_and_load_csv(file_content):
    """Parse CSV content matching training code logic"""
    try:
        raw = file_content.decode('utf-8').split('\n')
        cleaned = []
        
        for line in raw:
            s = line.strip()
            if not s:
                continue
            # Skip headers like ",CH1" or "Second,Volt" or "Time,Voltage"
            if s.startswith(",CH") or s.lower().startswith("second,") or s.lower().startswith("time,"):
                continue
            # Remove trailing commas
            s = s.rstrip(",")
            cleaned.append(s)
        
        if len(cleaned) < 5:
            raise ValueError("Not enough data rows")
        
        # Parse as Time,Voltage CSV
        df = pd.read_csv(StringIO("\n".join(cleaned)), header=None, names=["Time", "Voltage"])
        time = pd.to_numeric(df["Time"], errors="coerce")
        voltage = pd.to_numeric(df["Voltage"], errors="coerce")
        
        # Remove NaN values
        mask = (~time.isna()) & (~voltage.isna())
        time = time[mask].values
        voltage = voltage[mask].values
        
        if len(time) < 8:
            raise ValueError("Insufficient valid data points")
        
        return time, voltage
        
    except Exception as e:
        raise ValueError(f"Error parsing CSV file: {str(e)}")

def csv_to_image(time, voltage, target_size=(224, 224)):
    """Convert voltage data to image matching training code"""
    try:
        # Preprocess voltage (normalize like training)
        v_processed = preprocess_voltage(voltage)
        
        # Extract size (handle both int and tuple)
        img_size = target_size[0] if isinstance(target_size, tuple) else target_size
        
        # Create plot on WHITE background (same as training)
        fig = plt.figure(figsize=(3, 3), dpi=img_size//3)
        ax = plt.axes([0, 0, 1, 1])  # Full canvas, no margins
        
        # Plot waveform
        ax.plot(time, v_processed, linewidth=2)
        ax.set_axis_off()  # No axes (same as training)
        
        # Save to buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=img_size//3)
        plt.close(fig)
        buf.seek(0)
        
        # Load as PIL Image and resize to exact size
        img = Image.open(buf).convert('RGB')
        img = img.resize(target_size if isinstance(target_size, tuple) else (target_size, target_size), Image.BICUBIC)
        
        return img
    except Exception as e:
        raise ValueError(f"Error converting CSV to image: {str(e)}")

def get_plot_base64(time, voltage):
    """Generate detailed plot for frontend display - dark purple theme with data points"""
    try:
        # Apply smoothing for cleaner plot
        if HAS_SCIPY and len(voltage) > 50:
            v_smooth = savgol_filter(voltage, 51, 3)
        else:
            v_smooth = simple_smooth(voltage, window_size=5)
        
        # Create plot with dark purple background
        fig, ax = plt.subplots(figsize=(12, 5), facecolor='#2d1b4e')
        ax.set_facecolor('#2d1b4e')
        
        # Convert time to nanoseconds for better readability
        time_ns = time * 1e9
        
        # Plot the actual data points with markers
        ax.plot(time_ns, voltage, color='#3b9eff', linewidth=1.5, 
                marker='o', markersize=3, markerfacecolor='#3b9eff', 
                markeredgecolor='#ffffff', markeredgewidth=0.5,
                label='Voltage', alpha=0.9)
        
        # Styling to match the reference image
        ax.set_xlabel('Time (ns)', fontsize=12, color='white', fontweight='bold')
        ax.set_ylabel('Voltage (V)', fontsize=12, color='white', fontweight='bold')
        ax.set_title('Waveform Visualization', fontsize=14, color='white', fontweight='bold', pad=20)
        
        # Style the legend
        legend = ax.legend(loc='upper right', facecolor='#2d1b4e', edgecolor='#3b9eff', 
                          framealpha=0.8, fontsize=10)
        plt.setp(legend.get_texts(), color='#3b9eff')
        
        # Customize grid
        ax.grid(True, alpha=0.2, color='white', linestyle='-', linewidth=0.5)
        
        # Customize tick colors
        ax.tick_params(axis='x', colors='white', labelsize=10)
        ax.tick_params(axis='y', colors='white', labelsize=10)
        
        # Customize spines
        for spine in ax.spines.values():
            spine.set_color('white')
            spine.set_linewidth(1)
        
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100, facecolor='#2d1b4e')
        plt.close(fig)
        buf.seek(0)
        
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"
    except Exception as e:
        print(f"Error generating plot: {e}")
        return None

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_classifier_model():
    """Load the fault classification model (ResNet18)"""
    global classifier_model
    
    try:
        lazy_import_torchvision()
        
        model_path = os.path.join('models', 'image_fault_classifier.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Classifier model not found at {model_path}")
        
        # Create model (3 classes: Open, Short, Resistive)
        model = resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 3)
        
        # Load weights
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model = model.to(device)
        model.eval()
        
        classifier_model = model
        print(f"✓ Classifier model loaded successfully (device: {device})")
        return True
    except Exception as e:
        print(f"✗ Error loading classifier model: {e}")
        return False

def load_distance_regressor():
    """Load the hybrid distance regression model"""
    global distance_regressor, meta_ct
    
    try:
        lazy_import_torchvision()
        import joblib
        
        # Load the distance regressor model
        regressor_path = os.path.join('models', 'hybrid_distance_regressor.pth')
        if not os.path.exists(regressor_path):
            raise FileNotFoundError(f"Distance regressor not found at {regressor_path}")
        
        # Load meta transformer for preprocessing
        meta_ct_path = os.path.join('models', 'meta_ct.joblib')
        if not os.path.exists(meta_ct_path):
            raise FileNotFoundError(f"Meta transformer not found at {meta_ct_path}")
        
        # Create hybrid model architecture (1D CNN + metadata fusion)
        # Based on actual model weights:
        # - cnn.0: Conv1d(1, 32, kernel_size=7)
        # - cnn.3: Conv1d(32, 64, kernel_size=5)
        # - cnn.6: Conv1d(64, 128, kernel_size=3)
        # - head.0: Linear(132, 64)  <- 128 CNN features + 4 metadata features
        # - head.2: Linear(64, 1)
        class HybridDistanceRegressor(nn.Module):
            def __init__(self):
                super(HybridDistanceRegressor, self).__init__()
                # 1D CNN for waveform features
                self.cnn = nn.Sequential(
                    nn.Conv1d(1, 32, kernel_size=7, padding=3),  # cnn.0
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Conv1d(32, 64, kernel_size=5, padding=2),  # cnn.3
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Conv1d(64, 128, kernel_size=3, padding=1),  # cnn.6
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1)  # Global average pooling -> 128 features
                )
                
                # Regression head (CNN features + metadata)
                self.head = nn.Sequential(
                    nn.Linear(128 + 4, 64),  # head.0: 128 CNN + 4 metadata = 132
                    nn.ReLU(),
                    nn.Linear(64, 1)  # head.2: regression output
                )
            
            def forward(self, waveform, metadata):
                # waveform shape: (batch, 1, length)
                # metadata shape: (batch, 4)
                
                # Extract CNN features
                cnn_features = self.cnn(waveform)  # (batch, 128, 1)
                cnn_features = cnn_features.squeeze(-1)  # (batch, 128)
                
                # Concatenate with metadata
                combined = torch.cat([cnn_features, metadata], dim=1)  # (batch, 132)
                
                # Regression output
                distance = self.head(combined)  # (batch, 1)
                return distance
        
        # Load model
        model = HybridDistanceRegressor()
        model.load_state_dict(torch.load(regressor_path, map_location=device, weights_only=True))
        model = model.to(device)
        model.eval()
        distance_regressor = model
        
        # Load metadata transformer
        try:
            meta_ct = joblib.load(meta_ct_path)
            print(f"✓ Distance regressor loaded successfully (device: {device})")
            print(f"  Model uses 1D CNN + metadata fusion (VF, Z0, V, F)")
        except Exception as e:
            print(f"✗ Error loading meta_ct.joblib: {e}")
            print(f"  This will prevent distance prediction from working")
            distance_regressor = None
            meta_ct = None
            return False
        
        return True
    except Exception as e:
        print(f"✗ Error loading distance regressor: {e}")
        print(f"   Distance prediction will be disabled")
        return False

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    print("\n" + "="*60)
    print("TDR FAULT CLASSIFICATION BACKEND - STARTING UP")
    print("="*60)
    
    classifier_success = load_classifier_model()
    regressor_success = load_distance_regressor()
    
    if not classifier_success:
        print("\n⚠ WARNING: Classifier model failed to load!")
        print("API will start but fault classification will fail.\n")
    
    if not regressor_success:
        print("\n⚠ WARNING: Distance regressor failed to load!")
        print("Distance prediction will be disabled (fault classification still works).\n")
    
    if classifier_success and regressor_success:
        print("\n✓ All systems ready!")
        print("="*60 + "\n")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "message": "TDR Fault Classification API",
        "classifier_loaded": classifier_model is not None,
        "regressor_loaded": distance_regressor is not None,
        "device": device
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    velocityFactor: float = Form(0.67),
    characteristicImpedance: float = Form(50.0)
):
    """
    Perform fault classification and distance prediction on uploaded TDR data
    
    Args:
        file: CSV file with TDR waveform data
        velocityFactor: Velocity factor (VF) of the cable (default: 0.67, range 0-1)
        characteristicImpedance: Characteristic impedance Z0 in Ohms (default: 50)
    
    Returns: fault type classification, distance (if fault detected), and model outputs
    """
    try:
        # Check if classifier model is loaded
        if classifier_model is None:
            raise HTTPException(status_code=500, detail="Classifier model not loaded")
        
        # Read and parse CSV (returns time and voltage arrays)
        file_content = await file.read()
        time_data, voltage_data = clean_and_load_csv(file_content)
        
        if len(voltage_data) == 0:
            raise HTTPException(status_code=400, detail="No valid data in CSV file")
        
        # Convert CSV waveform to image (this is what the model sees!)
        # Training: Waveform → Image → ResNet18 → Classification
        img = csv_to_image(time_data, voltage_data, target_size=(224, 224))
        
        # Prepare image for classification (MUST match training transform exactly)
        lazy_import_torchvision()
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Ensure exact size
            transforms.ToTensor(),          # [0,1] range - NO normalization if training didn't use it
        ])
        
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # ============================================================================
        # STEP 1: FAULT CLASSIFICATION
        # ============================================================================
        with torch.no_grad():
            outputs = classifier_model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence_scores = probabilities[0].cpu().numpy()
            _, predicted = torch.max(outputs, 1)
            predicted_class = predicted.item()
        
        # Debug logging
        print(f"\n{'='*60}")
        print(f"DEBUG: Classification Results")
        print(f"{'='*60}")
        print(f"Raw logits: {outputs[0].cpu().tolist()}")
        print(f"Softmax probabilities: {confidence_scores.tolist()}")
        print(f"Predicted class: {predicted_class}")
        print(f"Max probability: {confidence_scores[predicted_class]:.4f}")
        print(f"{'='*60}\n")
        
        # Map prediction to fault type
        # Model outputs: 0=Open, 1=Short, 2=Resistive
        # UI expects: "No fault", "Short Circuit", "Resistive Fault"
        fault_mapping = {
            0: "No fault",           # Open circuit = No fault
            1: "Short Circuit",      # Short circuit
            2: "Resistive Fault"     # Resistive fault
        }
        
        fault_type = fault_mapping.get(predicted_class, "Unknown")
        
        # ============================================================================
        # STEP 2: DISTANCE PREDICTION (only if fault detected and regressor loaded)
        # ============================================================================
        distance_meters = None
        delta_t = None
        
        if predicted_class != 0 and distance_regressor is not None and meta_ct is not None:
            try:
                import pandas as pd
                
                # Prepare waveform for 1D CNN
                # The model expects: (batch, 1, length) - 1D signal
                preprocessed_voltage = preprocess_voltage(voltage_data)
                waveform_tensor = torch.tensor(preprocessed_voltage, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                # Shape: (1, 1, num_samples)
                
                # Prepare metadata features for regressor
                # Features: [vf, z0, V, F] - voltage and frequency might be in filename
                # For now, use defaults (you can extract from filename later)
                voltage = 3.0  # Default voltage
                frequency = 100.0  # Default frequency in kHz
                
                # Create metadata dataframe
                metadata_df = pd.DataFrame([[
                    velocityFactor,
                    characteristicImpedance,
                    voltage,
                    frequency
                ]], columns=['vf', 'z0', 'V', 'F'])
                
                # Transform metadata using the scaler
                metadata_transformed = meta_ct.transform(metadata_df)
                metadata_tensor = torch.tensor(metadata_transformed, dtype=torch.float32).to(device)
                # Shape: (1, 4)
                
                # Predict distance
                with torch.no_grad():
                    distance_pred = distance_regressor(waveform_tensor, metadata_tensor)
                    distance_meters = float(distance_pred.item())
                
                # Calculate delta_t (round-trip time)
                # delta_t = 2 * distance / (VF * c)
                # where c = 3e8 m/s (speed of light)
                speed_of_light = 3e8  # m/s
                delta_t = (2 * distance_meters) / (velocityFactor * speed_of_light)
                delta_t_ns = delta_t * 1e9  # Convert to nanoseconds
                
                print(f"✓ Distance prediction: {distance_meters:.2f} meters")
                print(f"  Delta-T: {delta_t_ns:.2f} ns")
                
            except Exception as e:
                print(f"⚠ Distance prediction failed: {e}")
                distance_meters = None
                delta_t = None
        elif predicted_class != 0 and distance_regressor is None:
            print(f"⚠ Distance prediction skipped: Regressor not loaded")
        
        # Generate plot for display (with time axis)
        plot_data = get_plot_base64(time_data, voltage_data)
        
        # Prepare detailed response (matching test code format)
        response = {
            "faultType": fault_type,
            "distance": distance_meters,  # Will be null if no fault or regressor not loaded
            "deltaT": delta_t,  # Round-trip time in seconds (null if no distance)
            "plotData": plot_data,
            "modelOutput": {
                "predicted_class": int(predicted_class),
                "predicted_fault_type": fault_type,
                "confidence_scores": {
                    "No fault (Open)": float(confidence_scores[0]),
                    "Short Circuit": float(confidence_scores[1]),
                    "Resistive Fault": float(confidence_scores[2])
                },
                "raw_logits": outputs[0].cpu().tolist(),
                "model_info": {
                    "device": str(device),
                    "input_shape": list(img_tensor.shape),
                    "num_samples": len(voltage_data),
                    "time_range": f"{time_data[0]:.6e} to {time_data[-1]:.6e} seconds"
                }
            }
        }
        
        print(f"✓ Classification: {fault_type} (class {predicted_class})")
        print(f"  Confidence: {confidence_scores[predicted_class]:.2%}")
        
        return JSONResponse(content=response)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"✗ Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("\nStarting server on http://localhost:8000")
    print("API docs available at http://localhost:8000/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
