import streamlit as st
from PIL import Image
import numpy as np
import keras

# App configuration
st.set_page_config(
    page_title="Casting Defect Detector",
    page_icon="⚙",
    layout="wide"
)

# Sidebar information
st.sidebar.header("About This Project")
st.sidebar.info("""
This application uses a deep learning model to detect manufacturing defects in cast metal parts.
The model is a Convolutional Neural Network (CNN) trained on the 'Casting Product Image Data' dataset using TensorFlow and Keras. 
It was initially trained using Google's Teachable Machine and deployed in this Streamlit app.
""")
st.sidebar.success("Project by: Daksh")

# Load model and labels
@st.cache_resource
def load_keras_model():
    labels_path = "labels.txt"
    model_path = "my_model"
    
    with open(labels_path, "r") as f:
        labels = [line.strip() for line in f.readlines()]
        
    model_layer = keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')
    return model_layer, labels

# Prediction function
def predict(image_to_predict, model_layer, labels):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    size = (224, 224)
    
    image = image_to_predict.resize(size)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    
    data[0] = normalized_image_array
    prediction_output = model_layer(data)
    prediction_tensor = list(prediction_output.values())[0]
    
    prediction = prediction_tensor.numpy()[0]
    index = np.argmax(prediction)
    class_name = labels[index]
    confidence_score = prediction[index]
    
    return class_name, confidence_score

# App title
st.title("Casting Defect Detection in Metal Impellers")

# Instructions
with st.expander("How to Use This App"):
    st.markdown("""
    1. **Upload an image** of a cast metal impeller.
    2. The AI model will analyze the image for manufacturing defects.
    3. The **Prediction** and **Confidence Score** will be shown.
    4. A result of **'Anomaly'** means a defect is likely.
    """)

# File uploader
uploaded_file = st.file_uploader("Upload an impeller image for inspection...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.header("Analysis Results")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_column_width=True)

    with col2:
        st.subheader("Prediction")
        with st.spinner("Analyzing the image..."):
            model_layer, labels = load_keras_model()
            class_name, confidence_score = predict(image, model_layer, labels)

        if class_name.lower() == "anomaly":
            st.error(f"Status: {class_name}")
            st.write(f"**Confidence:** {confidence_score:.2%}")
            st.warning("**Recommendation:** Manual inspection or rejection recommended.")
        else:
            st.success(f"Status: {class_name}")
            st.write(f"**Confidence:** {confidence_score:.2%}")
            st.info("**Recommendation:** Part passes automated inspection.")

else:
    st.header("Example Cases")
    st.write("No image uploaded yet. Below are examples of both normal and defective parts:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Normal Casting")
        st.image("ok_example.jpeg", caption="Example of a good part.")
    with col2:
        st.subheader("Defective Casting")
        st.image("def_example.jpeg", caption="Example of a casting anomaly.")
