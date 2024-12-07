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
    """Process the image to detect vehicles, extract license plates, and match the search number."""
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

    return matched_vehicles, boxes, original_image, plate_regions, plate_texts

def draw_annotations(image, boxes):
    """Draw bounding boxes around detected objects."""
    annotated_image = image.copy()
    for (x1, y1, x2, y2, cls, conf) in boxes:
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{COCO_CLASSES.get(cls, 'Unknown')} - {conf:.2f}"
        cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return annotated_image

# Streamlit UI
st.title("🚗 Vehicle Detection & License Plate Recognition")

# Instructions
st.markdown(
    """
    Upload an image containing vehicles to:
    - Detect vehicles using **YOLOv8**
    - Extract and recognize license plates
    - Search for a specific number in license plates
    """
)

# File Upload and Search Input
uploaded_file = st.file_uploader("📁 Upload an image (JPG/PNG)", type=["jpg", "png"])
search_number = st.text_input("🔍 Enter a partial or full vehicle number to search for")

if uploaded_file and search_number:
    # Read uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.write("🔄 **Processing the image...**")
    matches, boxes, original_image, plate_regions, plate_texts = process_image(image, search_number)

    # Display Annotated Image
    st.subheader("Step 1: Detected Vehicles with Bounding Boxes")
    annotated_image = draw_annotations(original_image, boxes)
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    st.image(annotated_image_rgb, caption="Annotated Image", use_column_width=True)

    # Display Cropped Plates and Extracted Text
    st.subheader("Step 2: Cropped License Plates & Extracted Text")
    for i, (plate, text) in enumerate(zip(plate_regions, plate_texts)):
        with st.container():
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB), caption=f"Plate {i+1}", use_column_width=True)
            with col2:
                st.markdown(f"**Extracted Text:** `{text}`")

    # Display Matched Vehicles
    st.subheader("Step 3: Matched Vehicles")
    if matches:
        for vehicle_type, plate_text, cropped_vehicle in matches:
            with st.container():
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.image(cv2.cvtColor(cropped_vehicle, cv2.COLOR_BGR2RGB), caption="Matched Vehicle", use_column_width=True)
                with col2:
                    st.markdown(f"**Vehicle Type:** {vehicle_type}")
                    st.markdown(f"**License Plate Text:** `{plate_text}`")
    else:
        st.info("No matching vehicles found.")
else:
    st.warning("Please upload an image and enter a search number to proceed.")
