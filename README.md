# Fish Freshness Analyzer

[![Heroku](https://img.shields.io/badge/Deployed-Heroku-blue)](https://fishfreshnessapp-dd83d72380b3.herokuapp.com/)

**Fish Freshness Analyzer** is a web-based application for analyzing the freshness of fish using image-based detection and colorimetric analysis. The application detects blue-bordered reference regions (black, white, and fish patch) in an image, computes corrected RGB, HSV, Chroma, Hue, and predicts freshness parameters such as pH, TVB-N, ammonia concentration, and remaining shelf life. Users can also generate a detailed PDF report of the analysis.

---

## 🖥️ Live Demo

Access the deployed app: [Fish Freshness Analyzer](https://fishfreshnessapp-dd83d72380b3.herokuapp.com/)

---

## 📂 File Structure

.
├── static/
│ ├── app.js
│ └── styles.css
├── templates/
│ └── index.html
├── application.py
├── requirements.txt
├── Procfile
└── README.md

markdown
Copy code

- **static/** – Contains CSS and JS files for frontend.  
- **templates/** – Contains HTML templates.  
- **application.py** – Main Flask application with image processing, freshness calculations, and PDF generation.  
- **requirements.txt** – Python dependencies.  
- **Procfile** – Deployment configuration for Heroku.  

---

## ⚙️ Features

- Detects blue-bordered reference regions in fish images.  
- Corrects RGB values using black and white reference regions.  
- Computes HSV, Hue, and Chroma values.  
- Calculates freshness parameters:  
  - pH  
  - TVB-N (Total Volatile Basic Nitrogen)  
  - Ammonia concentration ([NH3] + [NH4+])  
  - Remaining shelf life (days)  
- Generates a detailed PDF report with:  
  - Input parameters  
  - Extracted images  
  - Analysis results  
  - Warnings if parameters exceed limits  
- Real-time image processing via a simple web interface.  
- Cache management to store and clear the latest analysis.  

---

## 📥 Installation

1. Clone the repository:

```bash
git clone <repository_url>
cd fish-freshness-analyzer
Create and activate a virtual environment:

bash
Copy code
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Run the application locally:

bash
Copy code
python application.py
Access the app at:

cpp
Copy code
http://127.0.0.1:5000/
📄 Usage
Open the web app in your browser.

Upload an image of the fish with visible blue-bordered reference regions.

Enter the following input parameters:

Fish mass (g)

TVB-N limit (mg/100g)

Package volume (L)

Storage temperature (°C)

Click Analyze to view:

Corrected RGB, HSV, Chroma, Hue

Freshness parameters

Warnings (if any)

Download a detailed PDF report using Generate PDF.

🧪 Image Processing
Blue-bordered regions are detected using HSV masking and morphological operations.

RGB is corrected using black and white references.

Chroma and Hue are calculated to estimate freshness.

Shelf life is estimated based on temperature and colorimetric data.

🖨️ PDF Generation
PDF contains:

Input parameters

Extracted images

RGB, HSV, Chroma, Hue

Corrected values

Freshness results

Warnings

Generated using FPDF and downloadable.

🛠️ Dependencies
Key dependencies include:

Flask

OpenCV (cv2)

NumPy

SciPy

FPDF

Logging

Full list available in requirements.txt.

Install all dependencies with:

bash
Copy code
pip install -r requirements.txt
📡 Deployment
The app is deployed on Heroku.

Procfile specifies:

makefile
Copy code
web: python application.py
Environment variables:

app.secret_key is defined inside application.py.

⚠️ Notes
Ensure images contain exactly three blue-bordered regions (black, white, fish patch).

Cropped images are automatically validated; placeholders are used if detection fails.

Freshness parameters are based on calibrated colorimetric models and temperature.

📞 Contact
Developed by PACK Group, ATB Potsdam, Germany
Website: ATB Potsdam
