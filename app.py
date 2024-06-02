import cv2
import numpy as np
import streamlit as st
from pygame import mixer
from keras.models import load_model
from streamlit_option_menu import option_menu
import base64

# Load the trained model
model = load_model('Model.h5')

# Load the cascade classifiers
face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_eye.xml')

# Initialize audio mixer
mixer.init()
sound = mixer.Sound('alarm.mp3')

# Set up video capture
cap = cv2.VideoCapture(0)

# Initialize variables
Score = 0
alert_displayed = False

# Streamlit app
st.set_page_config(
    page_title="Car Care",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for aesthetics
css_string = """
    <style>
        .title {
            font-size: 2.5em;
            color: #333;
            text-align: center;
            margin-bottom: 0.5em;
            font-weight: bold;
            animation: fadeIn 2s ease-in-out;
        }
        .slider-label {
            font-size: 1.2em;
            color: #666;
            margin-bottom: 0.5em;
        }
        .stButton button {
            width: 100%;
            height: 3em;
            font-size: 1.2em;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .webrtc-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            margin-top: 1em;
            animation: fadeIn 2s ease-in-out;
        }
        .webrtc-video {
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        .sidebar .sidebar-content {
            padding-top: 50px;
        }
        .sidebar .sidebar-content h3 {
            font-size: 1.5em;
            color: #4CAF50;
            margin-bottom: 0.5em;
            font-weight: bold;
        }
        .sidebar .sidebar-content ul {
            list-style: none;
            padding-left: 0;
        }
        .sidebar .sidebar-content ul li {
            font-size: 1.2em;
            color: #333;
            margin-bottom: 0.5em;
            padding: 0.5em;
            cursor: pointer;
            border-radius: 5px;
            transition: background-color 0.3s ease;
        }
        .sidebar .sidebar-content ul li:hover {
            background-color: #f0f0f0;
        }
        .sidebar .sidebar-content ul li::before {
            content: "• ";
            color: #4CAF50;
            font-size: 1.5em;
            margin-right: 5px;
        }
        .option-menu {
            display: flex;
            justify-content: center;
            padding: 1em 0;
        }
        .side-block {
            margin-bottom: 15px;
        }
        .side-block img {
            width: 100%;
            max-width: 400px;
        }
        .side-block a.button {
            margin-bottom: 30px;
            background: #006CFF;
            color: #fff;
            font-weight: 500;
            font-size: 16px;
            line-height: 20px;
            padding: 15px;
            display: inline-block;
            max-width: 300px;
            border-radius: 5px;
            text-decoration: none;
        }
        .side-block a.button:hover {
            background: #000;
        }
        .centered {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            text-align: center;
        }
        .centered img {
            margin-bottom: 20px;
        }
    </style>
"""
st.markdown(css_string, unsafe_allow_html=True)

# Horizontal menu using streamlit-option-menu
selected = option_menu(
    menu_title=None,  # Optional: Title of the menu
    options=["Home", "App"],  # Options for the menu
    icons=["house", "app-indicator"],
    default_index=0,  # Optional: Default selected index
    orientation="horizontal",  # Horizontal orientation
)

# Helper function to get base64 encoded images
def get_base64_encoded_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    except Exception as e:
        st.error(f"Error loading image {image_path}: {e}")
        return None

# Define paths to your images
logo_image_path = "carlogo.jpeg"
ad_image_1_path = r"image1.jpeg"
ad_image_2_path = r"image2.jpeg"

# Encode images
logo_image_base64 = get_base64_encoded_image(logo_image_path)
ad_image_1_base64 = get_base64_encoded_image(ad_image_1_path)
ad_image_2_base64 = get_base64_encoded_image(ad_image_2_path)

if selected == "Home":
    if logo_image_base64:
        st.markdown(f'<div class="centered"><img src="data:image/jpeg;base64,{logo_image_base64}" width="400"></div>', unsafe_allow_html=True)
    st.markdown('<div class="centered"><h1>Welcome to the Car Care App</h1></div>', unsafe_allow_html=True)
    st.write("Here is some info about us.")
    st.markdown(f"""
        <div class="sidebar">
            <div class="side-block">
                <a target="_blank" href="https://your-link1.com" rel="noopener">
                    <img src="data:image/jpeg;base64,{ad_image_1_base64}" alt="Ad Image 1">
                </a>
            </div>
            <div class="side-block">
                <a target="_blank" href="https://www.instagram.com/car_care.app/" rel="noopener">
                    <img src="data:image/jpeg;base64,{ad_image_2_base64}" alt="Ad Image 2">
                </a>
            </div>
        </div>
    """, unsafe_allow_html=True)

elif selected == "App":
    st.markdown('<div class="title">Drowsiness Detection</div>', unsafe_allow_html=True)
    
    start_button = st.button('Start')
    stop_button = st.button('Stop')
    
    output_frame = st.empty()
    
    if start_button:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            height, width = frame.shape[0:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
            eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)
            
            cv2.rectangle(frame, (0, height-50), (200, height), (0, 0, 0), thickness=cv2.FILLED)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, pt1=(x, y), pt2=(x+w, y+h), color=(255, 0, 0), thickness=3)
                
            for (ex, ey, ew, eh) in eyes:
                eye = frame[ey:ey+eh, ex:ex+ew]
                eye = cv2.resize(eye, (80, 80))
                eye = eye / 255
                eye = eye.reshape(80, 80, 3)
                eye = np.expand_dims(eye, axis=0)
                
                prediction = model.predict(eye)
                
                if prediction[0][0] > 0.30:  # Closed eyes
                    cv2.putText(frame, 'closed', (10, height-20), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, color=(255, 255, 255),
                                thickness=1, lineType=cv2.LINE_AA)
                    cv2.putText(frame, 'Score: ' + str(Score), (100, height-20), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, color=(255, 255, 255),
                                thickness=1, lineType=cv2.LINE_AA)
                    Score += 1
                    if Score > 15 and not alert_displayed:
                        try:
                            sound.play()
                            st.warning("Eyes are closed!")
                            alert_displayed = True
                        except:
                            pass
                    
                elif prediction[0][1] > 0.90:  # Open eyes
                    cv2.putText(frame, 'open', (10, height-20), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, color=(255, 255, 255),
                                thickness=1, lineType=cv2.LINE_AA)      
                    cv2.putText(frame, 'Score: ' + str(Score), (100, height-20), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, color=(255, 255, 255),
                                thickness=1, lineType=cv2.LINE_AA)
                    Score -= 1
                    if Score < 0:
                        Score = 0
                    if alert_displayed:
                        st.warning("")
                        alert_displayed = False
            
            # Update the output frame with the processed frame
            output_frame.image(frame, channels='BGR')
            
            if stop_button:
                break
                
        cap.release()
        cv2.destroyAllWindows()
