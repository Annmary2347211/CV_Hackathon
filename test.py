import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import pytesseract

# Constants
COCO_CLASSES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

# Load YOLO Model
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')  # YOLOv8 Nano model

vehicle_model = load_model()

# Functions
def detect_vehicles(image):
    results = vehicle_model(image)
    detections = []
    for box in results[0].boxes.data:
        x1, y1, x2, y2, conf, cls = box.tolist()
        if int(cls) in COCO_CLASSES:
            detections.append((int(x1), int(y1), int(x2), int(y2), int(cls), conf))
    return detections, results[0].orig_img

def preprocess_plate(plate_image):
    """Preprocess the cropped plate for better OCR."""
    gray_plate = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    _, thresh_plate = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    resized_plate = cv2.resize(thresh_plate, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return resized_plate

def extract_text_from_plate(plate_image):
    """Extract text from a single plate image."""
    preprocessed_plate = preprocess_plate(plate_image)
    text = pytesseract.image_to_string(preprocessed_plate, config='--oem 3 --psm 6')
    return text.strip()

def process_image(image, search_number):
    """
    Process the image to detect vehicles, extract license plates, and match the search number.
    """
    boxes, original_image = detect_vehicles(image)
    plate_regions = [original_image[int(y1):int(y2), int(x1):int(x2)] for x1, y1, x2, y2, cls, conf in boxes]
    plate_texts = []
    matched_vehicles = []

    for i, plate_image in enumerate(plate_regions):
        text = extract_text_from_plate(plate_image)
        plate_texts.append(text)
        
        if search_number in text:  # Match the search number with the plate text
            x1, y1, x2, y2, cls, conf = boxes[i]
            cropped_vehicle = original_image[int(y1):int(y2), int(x1):int(x2)]
            matched_vehicles.append((COCO_CLASSES.get(cls, "Unknown"), text, cropped_vehicle))

    return matched_vehicles, boxes, original_image, plate_texts

def draw_annotations(image, boxes):
    """Draw bounding boxes around detected objects."""
    annotated_image = image.copy()
    for (x1, y1, x2, y2, cls, conf) in boxes:
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{COCO_CLASSES.get(cls, 'Unknown')} - {conf:.2f}"
        cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return annotated_image

# Streamlit UI
st.title("Vehicle Detection and License Plate Recognition")

# File Upload
uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "png"])

# Input Search Number
search_number = st.text_input("Enter a partial or full vehicle number to search for")

# Process Button
if uploaded_file and search_number:
    # Read uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    st.write("Processing the image...")
    
    # Process Image
    matches, boxes, original_image, plate_texts = process_image(image, search_number)

    st.subheader("Step 1: Detected Objects with Bounding Boxes")
    annotated_image = draw_annotations(original_image, boxes)
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    st.image(annotated_image_rgb, caption="Annotated Image", use_column_width=True)

    st.subheader("Step 2: Cropped License Plates and Extracted Text")
    plate_regions = [original_image[int(y1):int(y2), int(x1):int(x2)] for x1, y1, x2, y2, cls, conf in boxes]
    for i, (plate, text) in enumerate(zip(plate_regions, plate_texts)):
        st.image(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB), caption=f"License Plate {i+1}: {text}", use_column_width=True)

    st.subheader("Step 3: Detected and Matched Vehicles")
    if matches:
        for vehicle_type, plate_text, cropped_vehicle in matches:
            st.write(f"**Vehicle Type:** {vehicle_type}")
            st.write(f"**License Plate Text:** {plate_text}")
            st.image(cv2.cvtColor(cropped_vehicle, cv2.COLOR_BGR2RGB), caption="Matched Vehicle", use_column_width=True)
    else:
        st.write("No matching vehicles found.")

















"============================================================================================="
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
import re
from paddleocr import PaddleOCR

# Constants
COCO_CLASSES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

# Load YOLO Model
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')  # YOLOv8 Nano model

vehicle_model = load_model()

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # For English text detection

# Functions
def detect_vehicles(image):
    """Detect vehicles in the image using YOLO."""
    results = vehicle_model(image)
    detections = [
        (int(x1), int(y1), int(x2), int(y2), int(cls), conf)
        for x1, y1, x2, y2, conf, cls in results[0].boxes.data.tolist()
        if int(cls) in COCO_CLASSES
    ]
    return detections, results[0].orig_img

def preprocess_plate(plate_image):
    """Preprocess the cropped plate for better OCR."""
    gray_plate = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    thresh_plate = cv2.adaptiveThreshold(
        gray_plate, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    resized_plate = cv2.resize(thresh_plate, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return resized_plate

def extract_text_from_plate(plate_image):
    """Extract text from a single plate image using PaddleOCR."""
    if plate_image is None or plate_image.size == 0:
        return ""
    
    preprocessed_plate = preprocess_plate(plate_image)

    try:
        results = ocr.ocr(preprocessed_plate, cls=False)
        text = ' '.join(res[1][0] for res in results[0]) if results and results[0] else ""
    except Exception as e:
        print(f"Error in OCR: {e}")
        text = ""
    return clean_text(text)

def clean_text(text):
    """Remove unwanted characters and normalize spaces."""
    return re.sub(r'[^A-Za-z0-9\s]', '', text).strip()

def find_license_plate_region(image, boxes):
    """Locate license plate regions within detected vehicle bounding boxes."""
    plate_regions = []
    for x1, y1, x2, y2, cls, conf in boxes:
        vehicle_crop = image[int(y1):int(y2), int(x1):int(x2)]
        gray_vehicle = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
        _, thresh_vehicle = cv2.threshold(gray_vehicle, 150, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh_vehicle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 50 and h > 20:  # Filtering for plate-like dimensions
                plate_regions.append(vehicle_crop[y:y+h, x:x+w])
    return plate_regions

def process_image(image, search_number):
    """Process the image for vehicle detection, license plate extraction, and matching."""
    boxes, original_image = detect_vehicles(image)
    plate_regions = find_license_plate_region(original_image, boxes)
    plate_texts = [extract_text_from_plate(plate) for plate in plate_regions]
    matched_vehicles = [
        (COCO_CLASSES.get(cls, "Unknown"), text, original_image[int(y1):int(y2), int(x1):int(x2)])
        for (x1, y1, x2, y2, cls, conf), text in zip(boxes, plate_texts)
        if search_number in text
    ]
    return matched_vehicles, boxes, original_image, plate_regions, plate_texts

def draw_annotations(image, boxes):
    """Draw bounding boxes with labels on the image."""
    annotated_image = image.copy()
    for x1, y1, x2, y2, cls, conf in boxes:
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{COCO_CLASSES.get(cls, 'Unknown')} - {conf:.2f}"
        cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return annotated_image

# Streamlit UI
st.title("Vehicle Detection and License Plate Recognition")

uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "png"])
search_number = st.text_input("Enter a partial or full vehicle number to search for")

if uploaded_file and search_number:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.write("Processing the image...")

    matches, boxes, original_image, plate_regions, plate_texts = process_image(image, search_number)

    # Display annotated image
    st.subheader("Step 1: Detected Objects with Bounding Boxes")
    annotated_image = draw_annotations(original_image, boxes)
    st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), caption="Annotated Image", use_column_width=True)

    # Display cropped plates and extracted text
    st.subheader("Step 2: Cropped License Plates and Extracted Text")
    for plate, text in zip(plate_regions, plate_texts):
        st.image(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB), caption=f"Extracted Text: {text}", use_column_width=True)

    # Display matched vehicles
    st.subheader("Step 3: Detected and Matched Vehicles")
    if matches:
        for vehicle_type, plate_text, cropped_vehicle in matches:
            st.write(f"**Vehicle Type:** {vehicle_type}")
            st.write(f"**License Plate Text:** {plate_text}")
            st.image(cv2.cvtColor(cropped_vehicle, cv2.COLOR_BGR2RGB), caption="Matched Vehicle", use_column_width=True)
    else:
        st.write("No matching vehicles found.")
