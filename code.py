import streamlit as st
from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Initialize the VQA model and processor
model_name = "dandelin/vilt-b32-finetuned-vqa"
processor = ViltProcessor.from_pretrained(model_name)
model = ViltForQuestionAnswering.from_pretrained(model_name)

# Function to process and predict
def answer_question(image, question):
    inputs = processor(image, question, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    return model.config.id2label[idx]

# Function to capture image from webcam
class VideoProcessor(VideoTransformerBase):
    def _init_(self):
        self.frame = None

    def recv(self, frame):
        self.frame = frame.to_ndarray(format="bgr24")
        return frame

# Streamlit app
st.title("Visual Question Answering Bot")

# Sidebar for input selection
st.sidebar.title("Input Options")
input_option = st.sidebar.selectbox("Choose input method:", ["Upload an Image", "Use Webcam", "Image URL"])

# Image placeholder
image_placeholder = st.empty()

# Question input section
question = st.text_input("Ask a question about the image:")

# Image handling
if input_option == "Upload an Image":
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        image_placeholder.image(image, caption="Uploaded Image", use_column_width=True)

elif input_option == "Use Webcam":
    ctx = webrtc_streamer(key="example", video_processor_factory=VideoProcessor)
    if ctx.video_processor and ctx.video_processor.frame is not None:
        image = cv2.cvtColor(ctx.video_processor.frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image_placeholder.image(image, caption="Captured Image", use_column_width=True)

elif input_option == "Image URL":
    image_url = st.text_input("Enter Image URL:")
    if image_url:
        try:
            image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
            image_placeholder.image(image, caption="Image from URL", use_column_width=True)
        except:
            st.error("Invalid URL or unable to load image.")

# Submit button and processing
if st.button("Ask"):
    if question:
        if 'image' in locals():
            with st.spinner('Processing...'):
                try:
                    answer = answer_question(image, question)
                    st.write(f"Answer: {answer}")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.write("Please provide an image.")
    else:
        st.write("Please enter a question.")

# Sidebar information
st.sidebar.info("""
This app uses the ViLT model from Hugging Face to answer questions about images. 
You can upload an image, capture an image using your webcam, or provide an image URL.
""")
