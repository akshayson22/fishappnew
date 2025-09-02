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

app = Flask(__name__)
app.secret_key = 'fish_freshness_analyzer_secret_key'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants from the original code
PRIMARY_COLOR = '#1a2b3c'
SECONDARY_COLOR = '#0d1a26'
ACCENT_COLOR = '#3498db'
SUCCESS_COLOR = '#27ae60'
WARNING_COLOR = '#ff5555'
TEXT_COLOR = '#ffffff'

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
    shelf_life = max(0, ((corrected_hue - 130) / 20 * t_T)-0.5)

    # --- 4. Predict TVB-N based on Current Hue (New Linear Model) ---
    # Equation: TVB-N_pred = slope * corrected_hue + intercept
    #slope_tvbn = (30 - 15) / (130 - 150) # = 15 / (-20) = -0.75
    # Solve for intercept (b): 15 = (-0.75)*150 + b -> b = 15 + 112.5 = 127.5
    tvbn_pred = (-0.75) * corrected_hue + 127.5
    # Ensure prediction is within realistic bounds
    tvbn_pred = max(0, tvbn_pred)

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
    nh3_ppm_pred = max(0, nh3_ppm_pred)

    #####################

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
                placeholder = np.zeros((100, 100, 3), dtype=np.uint8)
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
                'ph': f"pH: {freshness_params['ph']:.2f}",
                'tvbn': f"TVB-N: {freshness_params['tvbn_pred']:.2f} mg/100g (limit: {tvbn_limit:.2f} mg/100g)",
                'ammonia': f"Ammonia: {freshness_params['nh3_ppm_pred']:.2f} ppm (limit: {freshness_params['nh3_ppm_limit']:.2f} ppm)",
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
    try:
        data = app.config.get('LAST_ANALYSIS')
        params = app.config.get('LAST_PARAMS')
        
        if not data:
            return jsonify({'error': 'No analysis available'}), 400

        pdf = FPDF()
        pdf.add_page()
        
        # Set margins and auto page break
        pdf.set_margins(15, 15, 15)
        pdf.set_auto_page_break(True, margin=15)
        
        # Header section - Uniform font sizes with link
        pdf.set_font("Arial", 'B', 18)
        pdf.cell(0, 12, "Fish Freshness Analyzer", 0, 1, 'C')
        pdf.set_font("Arial", 'I', 12)
        pdf.cell(0, 8, "Developed by PACK Group, ATB Potsdam, Germany", 0, 1, 'C', 
                link="https://www.atb-potsdam.de/en/about-us/areas-of-competence/systems-process-engineering/storage-and-packaging")
        pdf.ln(8)
        
        # Input Parameters section
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Input Parameters", 0, 1)
        pdf.set_font("Arial", '', 12)
        
        if params:
            pdf.cell(50, 8, "Fish Mass:", 0, 0)
            pdf.cell(0, 8, f"{params['fish_mass']:.2f} g", 0, 1)
            
            pdf.cell(50, 8, "TVB-N Limit:", 0, 0)
            pdf.cell(0, 8, f"{params['tvbn_limit']:.2f} mg/100g", 0, 1)
            
            pdf.cell(50, 8, "Temperature:", 0, 0)
            pdf.cell(0, 8, f"{params['temp_c']:.2f} °C", 0, 1)
            
            pdf.cell(50, 8, "Package Volume:", 0, 0)
            pdf.cell(0, 8, f"{params['headspace']:.2f} L", 0, 1)
        
        pdf.ln(10)
        
        # Extracted Images section
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Extracted Images", 0, 1)
        pdf.ln(5)
        
        # Save images to temporary files and add to PDF
        temp_files = []
        image_width = 55
        y_position = pdf.get_y()
        
        for i, (name, img_b64) in enumerate(data['images'].items()):
            label = {
                'original': 'Original Image',
                'black': 'Black Reference',
                'white': 'White Reference',
                'target': 'Fish Patch'
            }.get(name, name)
            
            # Check if we need a new page (every 2 images)
            if i % 2 == 0 and i > 0:
                pdf.add_page()
                y_position = pdf.get_y()
            
            # Calculate x position (left or right)
            x_position = 15 if i % 2 == 0 else 105
            
            # Set label with consistent font
            pdf.set_xy(x_position, y_position)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(image_width, 8, label, 0, 1)
            
            # Decode and save image to temporary file
            b64 = img_b64.split(',', 1)[1]
            imgdata = base64.b64decode(b64)
            tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            tmp.write(imgdata)
            tmp.flush()
            tmp.close()
            temp_files.append(tmp.name)
            
            # Add image with proper spacing
            try:
                pdf.image(tmp.name, x=x_position, y=pdf.get_y(), w=image_width)
            except Exception:
                logger.exception('Failed to insert image into PDF')
            
            # Update y position for next image
            if i % 2 == 0:
                # Same row, right side
                current_y = pdf.get_y()
                pdf.set_xy(x_position + image_width + 10, y_position + 8)
            else:
                # New row
                y_position = pdf.get_y() + 45
                pdf.set_xy(15, y_position)
        
        pdf.ln(50)
        
        # Analysis Results section
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Analysis Results", 0, 1)
        pdf.ln(5)
        
        # Reference Values - Use the actual data structure
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 8, "Reference Values", 0, 1)
        pdf.set_font("Arial", '', 11)
        
        # RGB Values
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 7, "RGB Values:", 0, 1)
        pdf.set_font("Arial", '', 11)
        rgb_text = data['results']['rgb'].replace("RGB - ", "")
        rgb_parts = rgb_text.split(' | ')
        for part in rgb_parts:
            pdf.cell(0, 6, part, 0, 1)
        pdf.ln(3)
        
        # HSV Values
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 7, "HSV Values:", 0, 1)
        pdf.set_font("Arial", '', 11)
        hsv_text = data['results']['hsv'].replace("HSV - ", "")
        hsv_parts = hsv_text.split(' | ')
        for part in hsv_parts:
            pdf.cell(0, 6, part, 0, 1)
        pdf.ln(3)
        
        # Chroma Values
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 7, "Chroma Values:", 0, 1)
        pdf.set_font("Arial", '', 11)
        chroma_text = data['results']['chroma'].replace("Chroma - ", "")
        chroma_parts = chroma_text.split(' | ')
        for part in chroma_parts:
            pdf.cell(0, 6, part, 0, 1)
        pdf.ln(3)
        
        # Hue Values
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 7, "Hue Values:", 0, 1)
        pdf.set_font("Arial", '', 11)
        hue_text = data['results']['hue'].replace("Hue - ", "")
        hue_parts = hue_text.split(' | ')
        for part in hue_parts:
            pdf.cell(0, 6, part, 0, 1)
        pdf.ln(5)
        
        # Corrected Values
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 8, "Corrected Values for Fish Patch", 0, 1)
        pdf.set_font("Arial", '', 11)
        
        corrected_data = [
            data['results']['corrected_rgb'],
            data['results']['corrected_v'],
            data['results']['corrected_chroma'],
            data['results']['corrected_hue']
        ]
        
        for line in corrected_data:
            if len(line) > 80:
                parts = [line[i:i+80] for i in range(0, len(line), 80)]
                for part in parts:
                    pdf.cell(0, 6, part, 0, 1)
            else:
                pdf.cell(0, 6, line, 0, 1)
            pdf.ln(2)
        
        pdf.ln(5)
        
        # Freshness Parameters
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 8, "Freshness Parameters", 0, 1)
        pdf.set_font("Arial", '', 11)
        
        freshness_data = [
            data['results']['ph'],
            data['results']['tvbn'],
            data['results']['ammonia'],
            data['results']['shelf_life']
        ]
        
        for line in freshness_data:
            pdf.cell(0, 6, line, 0, 1)
            pdf.ln(2)
        
        pdf.ln(5)
        
        # Warnings
        if data['results']['warnings']:
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 8, "Warnings", 0, 1)
            pdf.set_font("Arial", '', 11)
            
            for warning in data['results']['warnings']:
                if len(warning) > 80:
                    parts = [warning[i:i+80] for i in range(0, len(warning), 80)]
                    for part in parts:
                        pdf.cell(0, 6, part, 0, 1)
                else:
                    pdf.cell(0, 6, warning, 0, 1)
                pdf.ln(3)
        
        # Save PDF to temporary file
        out_temp = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
        pdf.output(out_temp.name)
        out_temp.flush()
        out_temp.close()
        
        # Cleanup temporary files
        for f in temp_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception:
                logger.exception("Failed to delete tmp file %s", f)
        
        return send_file(
            out_temp.name,
            as_attachment=True,
            download_name='analysis_report.pdf',
            mimetype='application/pdf'
        )
        
    except Exception as e:
        logger.exception("Error saving PDF")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
