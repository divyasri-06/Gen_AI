# Import the required Libraries
import streamlit as st
from PIL import Image
import pyttsx3
import pytesseract
from dotenv import load_dotenv
import os
import google.generativeai as genai
import numpy as np
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

# Set Tesseract OCR path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

load_dotenv()

# Initialize Generative AI with API Key
# f = open("keys\gemini.txt")
# key = f.read()
# config your Gemini API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# key = os.environ["Gemini_API_KEY"] 
# genai.configure(api_key=key)


# Streamlit Page Configuration

st.set_page_config(page_title="Vision AI ",layout="centered",page_icon=":eye:AI:")
st.title("Vision AI for Visually Impaired")

st.image("vision.jpeg")
st.sidebar.title("About ℹ️")
st.sidebar.markdown("An AI Solution for Visually Impaired People")
st.sidebar.markdown("""
 ** Features:**
 - _**Real-Time Scene Analysis**_: Description of Scene from uploaded image. 
 - _**Text to Speech Conversion**_: Converting text to audio description.
 - _**Object and Obstacle Detection**_: Detecting objects and obstacles for safe navigation.
 - _**Personalized Assistance**_: Provide task specific guidance.""")
 
st.markdown("""
_**How It Works**_:
- Upload an Image.
- Select a feature to interact with AI""")

# File Uploader

uploaded_file = st.file_uploader("Upload an image:", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
     st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

# Buttons

col1,col2,col3,col4 = st.columns(4)
scene_button = col1.button("Describe Scene 👁️")
txt_speech_button = col2.button("Extract Text 📝")
obj_obstacle_button = col3.button("Detect Objects 🔍")
assist_button = col4.button("Assist Task 🤖")
stop_button = st.button("Stop Audio")

# Prompt for AI

input_prompt = """
    You are an AI assistant to assist visually impaired individuals by analyzing images and providing descriptive outputs.
    Your task is to:
    - Analyze the uploaded image and describe its content in clear and simple language.
    - Keep description comprehensive and easy to understand.
    """

# Function to convert images to bytes

def input_image(uploaded_file):
     """Prepares the uploaded image for processing."""
     if uploaded_file is not None:
         bytes_data = uploaded_file.getvalue()
         image_parts = [
             {
                 "mime_type": uploaded_file.type,
                 "data": bytes_data,
             }
         ]
         return image_parts
     else:
         "No file uploaded"


# Function to get Scene Description

def get_scene(input_prompt, image_data):
    """Generate a scene description using Generative AI."""
    #model = genai.GenerativeModel("gemini-1.5-pro")
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    response = model.generate_content([input_prompt, image_data[0]])
    return response.text


# Function to extract text from image

def text_from_image(uploaded_file):
    """Extract the text from image using Tesseract OCR."""
    image = Image.open(uploaded_file)
    extracted_text = pytesseract.image_to_string(image)
    if not extracted_text.strip():
         return "No text Found"
    return extracted_text

# Function to convert text to speech

def text_to_speech(text):
    """Convert the text to speech using pyttsx3."""
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    engine.stop()

# Function to get assistance for the tasks

def get_assistance(input_prompt,image_data):
    """Generate task assisatnce using Generative AI."""
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content([input_prompt, image_data[0]])
    return response.text

# Function to detect Objects

def object_detection(image, model_path='yolov8n.pt'):
    # Load YOLO model
    model = YOLO(model_path)
    
    # Convert the image to OpenCV format (BGR)
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Perform object detection
    results = model(img)
    
    if results:
        # Get the first result (as the model returns a list of results)
        result = results[0]
        
        # Access the detections directly from result.boxes
        boxes = result.boxes  # This contains the bounding box data
        
        # Extract names and confidences
        names = boxes.cls.numpy()  # Class indices of the detected objects
        confidences = boxes.conf.numpy()  # Confidence scores of the detections
        
        class_names = model.names  # This provides the class names (list of names)
        
        detected_objects = [{'name': class_names[int(name)], 'confidence': confidence}
                            for name, confidence in zip(names, confidences)]
        
        # plot the bounding boxes on the image
        annotated_frame = result.plot() 
        
        return results, detected_objects, annotated_frame
    else:
        return None, None, None



if uploaded_file:
     
  # Display the uploaded image 
     
    image_data = input_image(uploaded_file)

     
    if txt_speech_button:
         with st.spinner(" Extracting text from image"):
             text = text_from_image(uploaded_file)
             st.write(text)
             if text.strip():
                text_to_speech(text)
                text_contents = text
                st.download_button("Download text", text_contents)

    if scene_button:
         with st.spinner(" Generate Scene Description"):
             response = get_scene(input_prompt,image_data)
             st.write(response)
             text_to_speech(response)
             text_contents = response
             st.download_button("Download text", text_contents)



    if obj_obstacle_button:
        with st.spinner("🔍 Recognizing objects in the image..."):
            image = Image.open(uploaded_file)
        
        # Convert image to RGB if necessary
            if image.mode != "RGB":
                image = image.convert("RGB")

        # Call object detection function
            results, detected_objects,annotated_frame = object_detection(image)
        
            if results is not None:
                st.success("✅ Objects Recognized Successfully!")
            
            # Convert annotated frame back to RGB for Streamlit
                annotated_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Display the annotated image
                st.image(annotated_image, caption="Objects Detected")
            
            # # Display object descriptions
                st.write("### Detected Objects:")
                for obj in detected_objects:
                    st.write(f"- **{obj['name']}** with confidence: {obj['confidence']:.2f}")
            else:
                st.error("❌ Error in recognizing objects.")
   
    if assist_button:
        with st.spinner("Providing task specific assistance"):
            assist_prompt = """"
            You are a helpful AI Assistant to help visually impaired individauls. 
            Analyse the uploaded image and identify tasks you can assist with as recognizing objects/obstacles, reading labels and providing safety insights if necessary in a simple and clear way."""
            response = get_assistance(assist_prompt,image_data)
            st.write(response)
            text_to_speech(response)
            text_contents = response
            st.download_button("Download text", text_contents)

    if stop_button:
         if "ts_engine" not in st.session_state:
             st.session_state.ts_engine = pyttsx3.init()
             st.session_state.ts_engine.stop()
             st.success("Audio stopped")

else:
    "No image uploaded yet"

#st.sidebar.markdown("Powered by Google API")



