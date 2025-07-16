# app.py
import cv2
import numpy as np
import os
import uuid
from flask import Flask, request, render_template, jsonify, url_for
from werkzeug.utils import secure_filename

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Configuration ---
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff'}
BLUR_THRESHOLD = 6600  # Default threshold for blur detection. Adjust as needed.

# Create the upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Core Computer Vision Functions ---

def analyze_blur(image_path, threshold):
    """
    Analyzes a single image for blurriness using the Laplacian variance method.
    This function is integrated from the original blur detector script.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None, None # Return None if image loading fails

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        variance = laplacian.var()
        is_blurry = variance < threshold
        return variance, is_blurry
    except Exception as e:
        print(f"Error in blur analysis: {e}")
        return None, None

def analyze_print_quality(template_path, scanned_path):
    """
    The main analysis function that performs both misalignment and blur checks.
    """
    try:
        # --- 1. Misalignment Analysis (from your original code) ---
        template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        scanned_img_gray = cv2.imread(scanned_path, cv2.IMREAD_GRAYSCALE)
        scanned_color = cv2.imread(scanned_path) # Load color version for output

        if template_img is None or scanned_img_gray is None:
            return None, "Error loading one or both images.", None, None

        # Feature Detection and Matching
        orb = cv2.ORB_create(2000)
        keypoints1, descriptors1 = orb.detectAndCompute(template_img, None)
        keypoints2, descriptors2 = orb.detectAndCompute(scanned_img_gray, None)

        if descriptors1 is None or descriptors2 is None:
            return None, "Could not find features in one or both images.", None, None

        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = sorted(matcher.match(descriptors1, descriptors2, None), key=lambda x: x.distance)
        num_good_matches = int(len(matches) * 0.15)
        matches = matches[:num_good_matches]

        # Image Alignment
        if len(matches) > 10:
            points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            h, _ = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)
            height, width = template_img.shape
            scanned_aligned = cv2.warpPerspective(scanned_img_gray, h, (width, height))
        else:
            scanned_aligned = scanned_img_gray # Fallback if alignment fails

        # Difference Analysis
        diff = cv2.absdiff(template_img, scanned_aligned)
        _, thresh_img = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        dilated_thresh = cv2.dilate(thresh_img, np.ones((5,5), np.uint8), iterations=2)
        contours, _ = cv2.findContours(dilated_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Highlight Errors
        misalignment_detected = False
        for contour in contours:
            if cv2.contourArea(contour) > 100:
                misalignment_detected = True
                (x, y, w, h) = cv2.boundingRect(contour)
                # Draw a red box on the color image
                cv2.rectangle(scanned_color, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        misalignment_message = "MISALIGNMENT DETECTED!" if misalignment_detected else "No significant misalignment found."

        # Save the output image with misalignment highlights
        output_filename = f"{uuid.uuid4().hex}_output.png"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        cv2.imwrite(output_path, scanned_color)

        # --- 2. Blur Analysis (on the original scanned image) ---
        blur_variance, is_blurry = analyze_blur(scanned_path, BLUR_THRESHOLD)

        return output_path, misalignment_message, blur_variance, is_blurry

    except Exception as e:
        return None, f"An unexpected error occurred: {str(e)}", None, None

# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    """Renders the main upload page."""
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_images():
    """Handles the image upload and processing logic."""
    if 'template' not in request.files or 'scanned' not in request.files:
        return jsonify({'error': 'Missing template or scanned image file.'}), 400

    template_file = request.files['template']
    scanned_file = request.files['scanned']

    if not (template_file and allowed_file(template_file.filename) and \
            scanned_file and allowed_file(scanned_file.filename)):
        return jsonify({'error': 'Invalid file type or no file selected.'}), 400

    # Save files with secure, unique names
    template_filename = secure_filename(f"{uuid.uuid4().hex}_{template_file.filename}")
    scanned_filename = secure_filename(f"{uuid.uuid4().hex}_{scanned_file.filename}")
    template_path = os.path.join(app.config['UPLOAD_FOLDER'], template_filename)
    scanned_path = os.path.join(app.config['UPLOAD_FOLDER'], scanned_filename)
    template_file.save(template_path)
    scanned_file.save(scanned_path)

    # Process the images
    output_path, mis_message, blur_score, is_blurry = analyze_print_quality(template_path, scanned_path)
    
    # Clean up the original uploaded files
    os.remove(template_path)
    os.remove(scanned_path)

    if output_path:
        return jsonify({
            'result_url': url_for('static', filename=f'uploads/{os.path.basename(output_path)}'),
            'misalignment_message': mis_message,
            'blur_score': f"{blur_score:.2f}" if blur_score is not None else "N/A",
            'is_blurry': bool(is_blurry) if is_blurry is not None else None,
            'blur_threshold': BLUR_THRESHOLD
        })
    else:
        return jsonify({'error': mis_message}), 500

if __name__ == '__main__':
    app.run(debug=True)
