# Fish Freshness Analyzer

[![Heroku](https://img.shields.io/badge/Deployed-Heroku-blue)](https://fishfreshnessapp-dd83d72380b3.herokuapp.com/)

A web app for analyzing fish freshness using image-based detection and colorimetric analysis. It predicts pH, TVB-N, ammonia concentration, and remaining shelf life, and generates detailed PDF reports.

---

## 🌐 Live Demo

[Try the app here](https://fishfreshnessapp-dd83d72380b3.herokuapp.com/)

---

## 📂 Structure

```
.
├── static/           # CSS & JS files
├── templates/        # HTML templates
├── application.py    # Main Flask app
├── requirements.txt  # Python dependencies
├── Procfile          # Heroku deployment config
└── README.md
```

---

## ⚙️ Features

* Detects blue-bordered reference regions (black, white, fish patch).
* Corrects RGB and computes HSV, Hue, Chroma.
* Calculates freshness: pH, TVB-N, ammonia, shelf life.
* Generates downloadable PDF reports with analysis and warnings.
* Real-time processing via web interface.

---

## 🚀 Quick Start

```bash
git clone <repository_url>
cd fish-freshness-analyzer
python -m venv venv
# Activate environment
# Linux/macOS
source venv/bin/activate
# Windows
venv\Scripts\activate
pip install -r requirements.txt
python application.py
```

Open in browser: `http://127.0.0.1:5000/`

---

## 🖼️ Usage

1. Upload fish image with blue-bordered regions.
2. Enter: fish mass, TVB-N limit, package volume, storage temperature.
3. Click **Analyze** → view corrected colors, freshness, warnings.
4. Generate PDF report.

---

## 🧪 Notes

* Images must contain exactly three blue-bordered regions.
* Freshness parameters are based on colorimetric calibration and temperature.

---

## 📞 Contact

**PACK Group, ATB Potsdam, Germany**
[https://www.atb-potsdam.de/](https://www.atb-potsdam.de/)
