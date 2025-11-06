# TDR Cable Fault Detection System

AI-powered cable fault detection using PyTorch ResNet18 + Hybrid CNN models.

## Quick Start

### 1. Install Frontend Dependencies

```bash
npm install
```

### 2. Setup Python Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 3. Add Your Trained Models

Place these files in `backend/models/`:
- `image_fault_classifier.pth` (ResNet18 classifier)
- `hybrid_distance_regressor.pth` (Hybrid regressor)
- `meta_ct.joblib` (Metadata transformer)

### 4. Run Backend (Terminal 1)

```bash
cd backend
venv\Scripts\activate
python main.py
```

Backend runs on: **http://localhost:8000**

### 5. Run Frontend (Terminal 2)

```bash
npm run dev
```

Frontend runs on: **http://localhost:3001**

### 6. Use the System

1. Open http://localhost:3001
2. Upload CSV file (Time, Voltage columns)
3. Enter VF (e.g., 0.67) and Z0 (e.g., 50)
4. Click "Analyze Cable"
5. View AI-powered results!

## File Structure

```
├── app/                    # Next.js frontend
│   ├── page.tsx           # Main dashboard
│   ├── layout.tsx
│   └── globals.css
├── components/            # React components
│   ├── FileUpload.tsx
│   ├── ParameterInput.tsx
│   ├── ResultsDisplay.tsx
│   └── WaveformChart.tsx
├── types/
│   └── tdr.ts
├── backend/
│   ├── main.py           # FastAPI + PyTorch models
│   ├── requirements.txt
│   └── models/           # ← PUT YOUR .pth FILES HERE
└── package.json
```

## How It Works

1. **CSV Upload** → Parse time/voltage data
2. **Image Generation** → Convert to 224×224 PNG
3. **ResNet18 CNN** → Classify fault (Open/Short/Resistive)
4. **Hybrid Model** → Predict distance (if not Open)
5. **Display Results** → Interactive charts + analysis

## Troubleshooting

**Backend won't start:**
- Make sure Python 3.8+ is installed
- Activate venv: `venv\Scripts\activate`
- Install PyTorch CPU version first

**Models not loading:**
- Check files in `backend/models/`
- Verify exact filenames match
- API will run with limited features if models missing

**Frontend connection error:**
- Ensure backend running on port 8000
- Check http://localhost:8000/health

## Tech Stack

- **Frontend:** Next.js 14, React 18, TypeScript, Tailwind CSS, Recharts
- **Backend:** FastAPI, PyTorch, NumPy, Pandas, Scikit-learn, Matplotlib
- **Models:** ResNet18 (classification) + Hybrid CNN (regression)
