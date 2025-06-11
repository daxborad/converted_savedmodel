import streamlit as st
from PIL import Image
import numpy as np
import keras
st.set_page_config(
    page_title="Casting Defect Detector",
    page_icon="⚙",
    layout="wide")
st.sidebar.header("About This Project")
st.sidebar.info("""
This application uses a deep learning model to detect manufacturing defects in cast metal parts.
The model is a Convolutional Neural Network (CNN) trained on the 'Casting Product Image Data' dataset using TensorFlow and Keras. It was initially trained using Google's Teachable Machine and the final model is deployed in this Streamlit application.""")
st.sidebar.success("Project by: Daksh")
@st.cache_resource
def load_keras_model():
    labels_path = "labels.txt"
    model_path = "my_model"
    with open(labels_path, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    model_layer = keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')
    return model_layer, labels
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
st.title("Cast Metal Impeller Anomaly Detection")

with st.expander("How to Use This App"):
    st.write("""
    1. **Upload an image** of a cast metal impeller using the file uploader.
    2. The AI model will analyze the image for common manufacturing defects.
    3. The **Status** and **Confidence Score** will be displayed on the right.
    4. **'Normal'** means the part has passed inspection. **'Anomaly'** means a defect was likely found.
   """)
uploaded_file = st.file_uploader("Upload an impeller image for inspection...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.header("Analysis Results")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Your Image")
        st.image(image, use_column_width=True)
    with col2:
        st.subheader("Prediction")
        with st.spinner('Analyzing the image...'):
            model_layer, labels = load_keras_model()
            class_name, confidence_score = predict(image, model_layer, labels)
        if class_name.lower() == "anomaly":
            st.error(f"Status: {class_name}")
            st.write(f"**Confidence:** {confidence_score:.2%}")
            st.warning("**Recommendation:** This part should be flagged for manual inspection or rejection.")
        else:
            st.success(f"Status: {class_name}")
            st.write(f"**Confidence:** {confidence_score:.2%}")
            st.info("**Recommendation:** This part has passed the automated inspection.")
else:
    st.header("Example Cases")
    st.write("No image uploaded yet. Check out these examples of normal and defective parts:")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Normal Casting")
        st.image("ok_example.jpeg", caption="A part that would pass inspection.")
    with col2:
        st.subheader("Anomaly (Defective)")
        st.image("def_example.jpeg", caption="A part with a casting defect.")

