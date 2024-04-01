import streamlit as st
from PIL import Image
from yolodemo.demo import get_labeled_image

st.title("Object Detection Demo")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Detecting objects...")

    # Save the uploaded image to a temporary file
    temp_file = f'temp_image.{uploaded_file.name.split(".")[-1]}'
    with open(temp_file, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Run object detection
    detected_image = get_labeled_image(temp_file)
    st.image(detected_image, caption="Detected Objects", use_column_width=True)
