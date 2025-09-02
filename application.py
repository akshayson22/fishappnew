import os
import base64
import tempfile
import math
import logging
import io
import cv2
import numpy as np
from fpdf import FPDF
from scipy import ndimage
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from io import BytesIO

app = Flask(__name__)
app.secret_key = 'fish_freshness_analyzer_secret_key'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Constants from the original code
PRIMARY_COLOR = '#1a2b3c'
SECONDARY_COLOR = '#0d1a26'
ACCENT_COLOR = '#3498db'
SUCCESS_COLOR = '#27ae60'
WARNING_COLOR = '#ff5555'
TEXT_COLOR = '#ffffff'

# ------------------------- Utility Functions -------------------------

def validate_image_b64(b64_string):
    """Validate base64 image string"""
    try:
        if not b64_string or ',' not in b64_string:
            return False
        header, enc = b64_string.split(',', 1)
        imgdata = base64.b64decode(enc)
        # Try to decode the image
        arr = np.frombuffer(imgdata, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img is not None and img.size > 0
    except Exception:
        return False

def cleanup_files(file_paths):
    """Safely remove temporary files"""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Successfully removed temporary file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to remove temporary file {file_path}: {str(e)}")

def create_placeholder_image(text="Image Error"):
    """Create a placeholder image when original is unavailable"""
    placeholder = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.putText(placeholder, text, (10, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return placeholder

def sanitize_text(text):
    """Replace Unicode characters with ASCII equivalents for PDF compatibility"""
    replacements = {
        '₃': '3',
        '°': ' degrees ',
        '±': '+/-',
        '×': 'x',
        '÷': '/',
        'α': 'alpha',
        'β': 'beta',
        'μ': 'mu',
        'Ω': 'Omega'
    }
    for unicode_char, ascii_char in replacements.items():
        text = text.replace(unicode_char, ascii_char)
    return text

# ------------------------- Image processing functions ----------------------

def detect_blue_bordered_regions(image):
    """
    Detect blue-bordered regions in an image (black, white, and fish patch)
    This is a direct port from the original AverageRGB.py code
    """
    try:
        processed = image.copy()
        height, width = processed.shape[:2]
        max_dim = max(height, width)
        scale = 800 / max_dim
        processed = cv2.resize(processed, None, fx=scale, fy=scale)
        original_scale = 1 / scale

        hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([100, 70, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        blue_mask = cv2.dilate(blue_mask, kernel, iterations=2)

        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:
                continue
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            if circularity < 0.2:
                continue
            valid_contours.append(cnt)

        if len(valid_contours) != 3:
            filled_mask = ndimage.binary_fill_holes(blue_mask).astype(np.uint8) * 255
            contours, _ = cv2.findContours(filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= 500]

        if len(valid_contours) != 3:
            return None

        valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)[:3]
        regions = []
        for contour in valid_contours:
            contour_mask = np.zeros_like(blue_mask)
            cv2.drawContours(contour_mask, [contour], -1, 255, -1)
            border_mask = cv2.bitwise_and(contour_mask, blue_mask)
            inner_mask = cv2.subtract(contour_mask, border_mask)
            inner_mask = ndimage.binary_fill_holes(inner_mask).astype(np.uint8) * 255
            x, y, w, h = cv2.boundingRect(contour)
            border_thickness = max(2, int(min(w, h) * 0.05))
            erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (border_thickness, border_thickness))
            inner_mask = cv2.erode(inner_mask, erosion_kernel, iterations=1)
            mean_color_bgr = cv2.mean(processed, mask=inner_mask)[:3]
            mean_color_rgb = (int(mean_color_bgr[2]), int(mean_color_bgr[1]), int(mean_color_bgr[0]))
            hsv_color = cv2.cvtColor(np.array([[mean_color_bgr]], dtype=np.uint8), cv2.COLOR_BGR2HSV)[0, 0]
            S = hsv_color[1] / 2.55
            V = hsv_color[2] / 2.55
            chroma = (S / 100) * V
            hue = float(hsv_color[0]) * 2.0
            if hue > 360:
                hue = hue - 360
            inner_region = cv2.bitwise_and(processed, processed, mask=inner_mask)
            cropped = inner_region[y:y+h, x:x+w]
            if scale != 1.0:
                cropped = cv2.resize(cropped, None, fx=original_scale, fy=original_scale)

            # Robust trimming via erosion
            erosion_margin = 10
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_margin, erosion_margin))
            robust_mask = cv2.erode(inner_mask, kernel, iterations=1)
            robust_cropped = cv2.bitwise_and(processed, processed, mask=robust_mask)
            x, y, w, h = cv2.boundingRect(robust_mask)
            cropped = robust_cropped[y:y+h, x:x+w]

            regions.append({
                "rgb": mean_color_rgb,
                "hsv": (round(hsv_color[0], 1), round(hsv_color[1] / 2.55, 1), round(hsv_color[2] / 2.55, 1)),
                "chroma": round(chroma, 1),
                "hue": round(hue, 1),
                "cropped": cropped,
                "avg_V": V
            })

        regions_sorted = sorted(regions, key=lambda x: x['avg_V'])
        return {"black": regions_sorted[0], "target": regions_sorted[1], "white": regions_sorted[2]}

    except Exception as e:
        logger.exception('Error in detect_blue_bordered_regions')
        return None


def rgb_to_hsv_chroma_hue(r, g, b):
    """Convert RGB to HSV, Chroma, and Hue manually"""
    r /= 255.0
    g /= 255.0
    b /= 255.0

    c_max = max(r, g, b)
    c_min = min(r, g, b)
    delta = c_max - c_min

    # Hue calculation
    if delta == 0:
        hue = 0
    elif c_max == r:
        hue = (60 * ((g - b) / delta)) % 360
    elif c_max == g:
        hue = (60 * ((b - r) / delta)) + 120
    elif c_max == b:
        hue = (60 * ((r - g) / delta)) + 240

    # Value
    value = c_max

    # Saturation
    saturation = 0 if c_max == 0 else delta / c_max

    # Chroma (value * saturation)
    chroma = saturation * value

    return hue, chroma, value


def calculate_freshness_parameters(corrected_rgb, corrected_hue, corrected_chroma, corrected_value, 
                                 tvbn_limit, fish_mass, headspace, temp_c):
    """Calculate freshness parameters based on the original code"""
    
    # pH calculation
    # --- 1. NEW pH Calculation based on provided Hue-pH correlation ---
    ph_from_hue = 10.6 - (0.03 * corrected_hue)
    avg_ph = ph_from_hue

    # --- 2. Calculate Spoilage Rate and Total Shelf Life at current temp using Arrhenius ---
    # Constants from our derived model
    A = 5.35e13  # Pre-exponential factor (day⁻¹)
    Ea = 77850   # Activation Energy (J/mol)
    R = 8.314    # Gas Constant (J/mol·K)

    # Convert temperature to Kelvin
    temp_k = temp_c + 273.15

    # Calculate the spoilage rate r(T) and total shelf life t(T)
    r_T = A * math.exp(-Ea / (R * temp_k)) # Spoilage rate (day⁻¹)
    t_T = 1 / r_T                          # Total shelf life (days) from H=150 to H=130

    # --- 3. Calculate Remaining Shelf Life based on Current Hue ---
    shelf_life = max(0, (corrected_hue - 130) / 20 * t_T)

    # --- 4. Predict TVB-N based on Current Hue (New Linear Model) ---
    # Equation: TVB-N_pred = slope * corrected_hue + intercept
    #slope_tvbn = (30 - 15) / (130 - 150) # = 15 / (-20) = -0.75
    # Solve for intercept (b): 15 = (-0.75)*150 + b -> b = 15 + 112.5 = 127.5
    tvbn_pred = (-0.75) * corrected_hue + 127.5
    # Ensure prediction is within realistic bounds
    tvbn_pred = max(0, min(30, tvbn_pred))

    # --- 5. (Optional) Keep your original NH3/Headspace calculation for limits ---
    # This section is kept for compatibility but is no longer used for shelf life.
    # Calculate ammonia limit based on input parameters
    P = 1  # Pressure (atm)
    molecular_weight = 17.03  # NH₃
    molar_volume = (R * temp_k) / P # Uses R from Arrhenius calculation above

    # TVB-N calculation using the user-defined tvbn_limit
    tvbn_mg = (tvbn_limit * fish_mass) / 100

    # Compute gaseous NH3 fraction (temperature/pH-dependent)
    total_nh3_mg = 0.6 * tvbn_mg  # Assuming that 60% of TVB-N is Ammonia

    pKa = 0.09018 + (2729.92 / temp_k)  # Temperature-dependent pKa
    NH3_ratio = 10 ** (ph_from_hue - pKa)  # [NH3]/[NH4+]
    f_NH3 = NH3_ratio / (1 + NH3_ratio)  # Fraction of free NH3
    gaseous_nh3_mg = f_NH3 * total_nh3_mg  # Free and gaseous ammonia

    tvbn_mg_current = (tvbn_pred * fish_mass) / 100 # Predicted TVB-N mass
    total_nh3_mg_current = 0.6 * tvbn_mg_current  # Assume 60% of TVB-N is Ammonia
    gaseous_nh3_mg_current = f_NH3 * total_nh3_mg_current # Gaseous ammonia mass now
                                     
    # Fish volume and headspace
    density_kg_per_m3 = 1080.0
    fish_volume_L = ((fish_mass / 1000) / density_kg_per_m3) * 1000
    Actual_headspace = max(headspace - fish_volume_L, 0.001)

    # Convert gaseous NH₃ to ppm (this is the safety limit)
    nh3_ppm_limit = (gaseous_nh3_mg * molar_volume) / (molecular_weight * Actual_headspace)

    # --- 6. Empirical NH3 prediction from Chroma ---
    nh3_ppm_pred = (gaseous_nh3_mg_current * molar_volume) / (molecular_weight * Actual_headspace)
    nh3_ppm_pred = max(0, min(100, nh3_ppm_pred))

    ########################################################################

    # Validate results
    warnings = []

    if avg_ph < 4 or avg_ph > 14:
        warnings.append(f"pH value out of range (4-14): {avg_ph:.2f}")
    if tvbn_pred < 0 or tvbn_pred > 100:
        warnings.append(f"TVB-N value out of range (0-100 mg/100g): {tvbn_pred:.2f}")
    if tvbn_pred > tvbn_limit:
        warnings.append(f"High TVB-N ({tvbn_pred:.2f} mg/100g > limit {tvbn_limit:.2f} mg/100g) - potentially spoiled")
    if nh3_ppm_pred < 0 or nh3_ppm_pred > 100:
        warnings.append(f"Ammonia value out of range (0-100 ppm): {nh3_ppm_pred:.2f}")
    if nh3_ppm_pred > nh3_ppm_limit:
        warnings.append(f"High ammonia concentration ({nh3_ppm_pred:.2f} ppm > limit {nh3_ppm_limit:.2f} ppm) - potentially spoiled")
    if shelf_life < 0 or shelf_life > 20:
        warnings.append(f"Shelf life out of range (0-20 days): {shelf_life:.2f}")
    if shelf_life < 1:
        warnings.append("Very short shelf life (less than 1 day) - consume immediately")

    return {
        'ph': round(avg_ph, 2),
        'tvbn_pred': round(tvbn_pred, 2),
        'nh3_ppm_pred': round(nh3_ppm_pred, 1),
        'nh3_ppm_limit': round(nh3_ppm_limit, 1),
        'shelf_life': round(shelf_life, 1),
        'warnings': warnings
    }


def b64_to_cv2(b64str):
    """Convert base64 string to OpenCV image"""
    header, enc = b64str.split(',', 1)
    data = base64.b64decode(enc)
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def cv2_to_b64(img):
    """Convert OpenCV image to base64 string"""
    _, buf = cv2.imencode('.png', img)
    b64 = base64.b64encode(buf).decode('ascii')
    return 'data:image/png;base64,' + b64


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process_image():
    try:
        data = request.json
        image_data = data['image']
        tvbn_limit = float(data['tvbn_limit'])
        fish_mass = float(data['fish_mass'])
        headspace = float(data['headspace'])
        temp_c = float(data['temp'])

        # Convert base64 to OpenCV image
        img = b64_to_cv2(image_data)
        
        # Detect blue-bordered regions
        result = detect_blue_bordered_regions(img)
        
        if result is None:
            return jsonify({'error': 'Failed to detect exactly 3 blue-bordered regions'})

        info = result

        # Extract RGB values (as float32 for precision)
        black_rgb = np.array(info['black']['rgb'], dtype=np.float32)
        white_rgb = np.array(info['white']['rgb'], dtype=np.float32)
        target_rgb = np.array(info['target']['rgb'], dtype=np.float32)

        # Correct RGB using black and white references
        # Avoid division by zero by adding a small epsilon
        epsilon = 1e-6
        corrected_rgb = (target_rgb - black_rgb) / (white_rgb - black_rgb + epsilon)
        corrected_rgb = np.clip(corrected_rgb, 0, 1) * 255  # scale to [0, 255]

        # Convert corrected RGB to HSV manually and compute chroma and hue
        corrected_hue, corrected_chroma_raw, corrected_value = rgb_to_hsv_chroma_hue(*corrected_rgb)

        # Scale chroma to 0-100 to match reference scale
        corrected_chroma = corrected_chroma_raw * 100

        # Calculate freshness parameters
        freshness_params = calculate_freshness_parameters(
            corrected_rgb, corrected_hue, corrected_chroma, corrected_value,
            tvbn_limit, fish_mass, headspace, temp_c
        )

        # Prepare response with image validation
        images_response = {
            'original': cv2_to_b64(img)
        }
        
        # Add cropped images only if they are not empty
        for name in ['black', 'white', 'target']:
            cropped_img = info[name]['cropped']
            # Check if the cropped image is not empty
            if cropped_img is not None and cropped_img.size > 0:
                images_response[name] = cv2_to_b64(cropped_img)
            else:
                # Create a placeholder black image
                placeholder = create_placeholder_image(f"{name.capitalize()} Not Found")
                images_response[name] = cv2_to_b64(placeholder)
                logger.warning(f"Empty cropped image for {name}, using placeholder")

        response = {
            'images': images_response,
            'results': {
                'rgb': f"RGB - Black: {info['black']['rgb'][0]}, {info['black']['rgb'][1]}, {info['black']['rgb'][2]} | White: {info['white']['rgb'][0]}, {info['white']['rgb'][1]}, {info['white']['rgb'][2]} | Fish Patch: {info['target']['rgb'][0]}, {info['target']['rgb'][1]}, {info['target']['rgb'][2]}",
                'hsv': f"HSV - Black: {info['black']['hsv'][0]:.2f}, {info['black']['hsv'][1]:.2f}, {info['black']['hsv'][2]:.2f} | White: {info['white']['hsv'][0]:.2f}, {info['white']['hsv'][1]:.2f}, {info['white']['hsv'][2]:.2f} | Fish Patch: {info['target']['hsv'][0]:.2f}, {info['target']['hsv'][1]:.2f}, {info['target']['hsv'][2]:.2f}",
                'chroma': f"Chroma - Black: {info['black']['chroma']:.2f} | White: {info['white']['chroma']:.2f} | Fish Patch: {info['target']['chroma']:.2f}",
                'hue': f"Hue - Black: {info['black']['hue']:.2f}° | White: {info['white']['hue']:.2f}° | Fish Patch: {info['target']['hue']:.2f}°",
                'corrected_rgb': f"Corrected RGB: {int(corrected_rgb[0])}, {int(corrected_rgb[1])}, {int(corrected_rgb[2])}",
                'corrected_v': f"Corrected V (Value): {corrected_value:.2f}",
                'corrected_chroma': f"Corrected Chroma: {corrected_chroma:.2f}",
                'corrected_hue': f"Corrected Hue: {corrected_hue:.2f}°",
                'ph': f"pH: {freshness_params['ph']:.2f} (±0.45)",
                'tvbn': f"TVB-N: {freshness_params['tvbn_pred']:.2f} mg/100g (limit: {tvbn_limit:.2f} mg/100g)",
                'ammonia': f"Ammonia (NH₃ + NH₄⁺): {freshness_params['nh3_ppm_pred']:.2f} ppm (limit: {freshness_params['nh3_ppm_limit']:.2f} ppm)",
                'shelf_life': f"Remaining Shelf Life: {freshness_params['shelf_life']:.2f} days",
                'warnings': freshness_params['warnings']
            }
        }

        # Store for PDF generation
        app.config['LAST_ANALYSIS'] = response
        app.config['LAST_PARAMS'] = {
            'tvbn_limit': tvbn_limit,
            'fish_mass': fish_mass,
            'headspace': headspace,
            'temp_c': temp_c
        }

        return jsonify(response)

    except Exception as e:
        logger.exception("Error during processing")
        return jsonify({'error': str(e)}), 500


@app.route('/save_pdf', methods=['POST'])
def save_pdf():
    temp_files = []
    out_temp = None
    
    try:
        logger.info("PDF generation started")
        
        data = app.config.get('LAST_ANALYSIS')
        params = app.config.get('LAST_PARAMS')
        
        if not data:
            logger.error("No analysis data available for PDF generation")
            return jsonify({'error': 'No analysis available'}), 400

        pdf = FPDF()
        pdf.add_page()
        
        # Use core font to avoid Unicode issues
        pdf.set_font("Helvetica", size=10)
        
        # Header
        pdf.set_font("Helvetica", 'B', 16)
        pdf.cell(200, 10, text="Fish Freshness Analyzer", align='C')
        pdf.ln(8)
        pdf.set_font("Helvetica", '', 10)
        pdf.cell(200, 8, text="Developed by PACK Group, ATB Potsdam, Germany", align='C')
        pdf.ln(12)
        
        # Input Parameters section
        pdf.set_font("Helvetica", 'B', 14)
        pdf.cell(200, 10, text="Input Parameters")
        pdf.ln(8)
        pdf.set_font("Helvetica", '', 10)
        
        if params:
            pdf.cell(60, 8, text="Fish Mass:")
            pdf.cell(0, 8, text=f"{params['fish_mass']:.2f} g")
            pdf.ln(8)
            
            pdf.cell(60, 8, text="TVB-N Limit:")
            pdf.cell(0, 8, text=f"{params['tvbn_limit']:.2f} mg/100g")
            pdf.ln(8)
            
            pdf.cell(60, 8, text="Temperature:")
            pdf.cell(0, 8, text=f"{params['temp_c']:.2f} C")  # Removed ° symbol
            pdf.ln(8)
            
            pdf.cell(60, 8, text="Package Volume:")
            pdf.cell(0, 8, text=f"{params['headspace']:.2f} L")
            pdf.ln(8)
        
        pdf.ln(10)
        
        # Analysis Results section - Extract only the numeric values
        pdf.set_font("Helvetica", 'B', 14)
        pdf.cell(200, 10, text="Analysis Results")
        pdf.ln(8)
        
        # Extract only numeric values from the results to avoid text parsing issues
        try:
            # Parse the numeric values from the text strings
            def extract_value(text_string, prefix):
                """Extract numeric value from formatted text"""
                if prefix in text_string:
                    # Find the numeric part after the prefix
                    parts = text_string.split(prefix)
                    if len(parts) > 1:
                        # Extract first number found
                        import re
                        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", parts[1])
                        if numbers:
                            return float(numbers[0])
                return "N/A"
            
            # Extract values safely
            ph_value = extract_value(data['results']['ph'], "pH: ")
            tvbn_value = extract_value(data['results']['tvbn'], "TVB-N: ")
            ammonia_value = extract_value(data['results']['ammonia'], "Ammonia (NH3 + NH4+): ")
            shelf_life_value = extract_value(data['results']['shelf_life'], "Remaining Shelf Life: ")
            
            # Freshness Parameters
            pdf.set_font("Helvetica", 'B', 12)
            pdf.cell(200, 8, text="Freshness Parameters")
            pdf.ln(8)
            pdf.set_font("Helvetica", '', 10)
            
            # Add key results using extracted numeric values
            pdf.cell(80, 8, text="pH:")
            pdf.cell(0, 8, text=f"{ph_value:.2f} (+/- 0.45)" if ph_value != "N/A" else "N/A")
            pdf.ln(8)
            
            pdf.cell(80, 8, text="TVB-N:")
            pdf.cell(0, 8, text=f"{tvbn_value:.2f} mg/100g" if tvbn_value != "N/A" else "N/A")
            pdf.ln(8)
            
            pdf.cell(80, 8, text="Ammonia (NH3 + NH4+):")
            pdf.cell(0, 8, text=f"{ammonia_value:.2f} ppm" if ammonia_value != "N/A" else "N/A")
            pdf.ln(8)
            
            pdf.cell(80, 8, text="Shelf Life:")
            pdf.cell(0, 8, text=f"{shelf_life_value:.2f} days" if shelf_life_value != "N/A" else "N/A")
            pdf.ln(8)
            
        except Exception as e:
            logger.error(f"Error extracting values: {str(e)}")
            pdf.multi_cell(0, 8, text="Error: Could not extract analysis values from results.")
            pdf.ln(8)
        
        # Warnings - sanitize thoroughly
        if data['results']['warnings']:
            pdf.set_font("Helvetica", 'B', 12)
            pdf.cell(200, 10, text="Warnings")
            pdf.ln(8)
            pdf.set_font("Helvetica", '', 10)
            for warning in data['results']['warnings']:
                # Comprehensive sanitization
                safe_warning = warning
                # Remove all non-ASCII characters
                safe_warning = ''.join(char for char in safe_warning if ord(char) < 128)
                # Additional replacements
                safe_warning = safe_warning.replace('°', ' degrees ')
                safe_warning = safe_warning.replace('±', '+/-')
                safe_warning = safe_warning.replace('×', 'x')
                safe_warning = safe_warning.replace('÷', '/')
                
                try:
                    pdf.multi_cell(0, 8, text=safe_warning)
                except:
                    # Ultimate fallback: ASCII-only
                    ascii_warning = safe_warning.encode('ascii', 'ignore').decode('ascii')
                    pdf.multi_cell(0, 8, text=ascii_warning)
                pdf.ln(5)
        
        # Save PDF to temporary file in upload folder
        upload_folder = app.config.get('UPLOAD_FOLDER', '/tmp')
        os.makedirs(upload_folder, exist_ok=True)
        
        out_temp = tempfile.NamedTemporaryFile(
            suffix='.pdf', 
            delete=False,
            dir=upload_folder
        )
        pdf.output(out_temp.name)
        out_temp.flush()
        
        logger.info("PDF generated successfully")
        
        # Send file response
        return send_file(
            out_temp.name,
            as_attachment=True,
            download_name=secure_filename('analysis_report.pdf'),
            mimetype='application/pdf'
        )
        
    except Exception as e:
        logger.exception("Detailed error in save_pdf:")
        return jsonify({'error': f'PDF generation failed: {str(e)}'}), 500
        
    finally:
        # Cleanup temporary files
        cleanup_files(temp_files)
        if out_temp and os.path.exists(out_temp.name):
            cleanup_files([out_temp.name])


if __name__ == '__main__':
    # Verify directory permissions
    upload_folder = app.config.get('UPLOAD_FOLDER', '/tmp')
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    if not os.access(upload_folder, os.W_OK):
        logger.error(f"No write permission for directory: {upload_folder}")
    
    logger.info(f"Starting Flask application with upload folder: {upload_folder}")
    app.run(debug=True, host='0.0.0.0', port=5000)
