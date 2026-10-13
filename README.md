# Pneumonia X-ray classification (research demonstration)

Use Python 3.10+ on Windows. In PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py --epochs 5
```

Or supply an existing extracted dataset directory with train/, val/, test/:

```powershell
python main.py --data-dir "C:\\path\\to\\chest_xray" --epochs 5
```

Evaluate an existing model without retraining:

```powershell
python main.py --eval-only --image "C:\\path\\to\\xray.jpeg"
```

Results are saved to outputs/. The first run downloads the Kaggle dataset and pretrained DenseNet weights if not cached. Grad-CAM is a qualitative visualization, not a validated medical explanation. This is an educational model, not a diagnostic device. The original validation split is tiny; reported metrics should not be interpreted as clinical performance.
