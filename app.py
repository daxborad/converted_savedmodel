import streamlit as st
from PIL import Image
import numpy as np
import keras

# --- PAGE CONFIGURATION ---
# Sets the title and icon that appear in the browser tab
st.set_page_config(
    page_title="Welding Anomaly Detector",
    page_icon="🔥",
    layout="wide"
)

# --- SIDEBAR FOR PROJECT INFORMATION ---
st.sidebar.header("About This Project")
st.sidebar.info("""
    This application uses a deep learning model to detect anomalies in welded joints.

    The model is a Convolutional Neural Network (CNN) trained on a dataset of welding images using TensorFlow and Keras. It was created with Google's Teachable Machine and deployed in this Streamlit application.
""")
st.sidebar.success("Project by: Ashvin (Modified by AI)")


# --- MODEL LOADING ---
@st.cache_resource
def load_keras_model():
    """
    Loads the Keras model and labels from the disk.
    The model is loaded as a TFSMLayer, which is the Keras 3 way for SavedModels.
    """
    # IMPORTANT: Make sure these file paths are correct for your project
    labels_path = "labels.txt"
    model_path = "keras_model.h5"  # Or your model's folder name if it's a SavedModel directory

    # Load the labels
    with open(labels_path, "r") as f:
        labels = [line.strip() for line in f.readlines()]

    # Load the model
    # Note: If your model is a .h5 file, you might need to load it differently, e.g., with keras.models.load_model()
    # This example assumes a TensorFlow SavedModel format, which is common with Teachable Machine.
    model_layer = keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')

    return model_layer, labels

# --- PREDICTION FUNCTION ---
def predict(image_to_predict, model_layer, labels):
    """
    Takes a PIL image and a TFSMLayer, and returns the predicted class and confidence score.
    """
    # Create the array of the right shape to feed into the model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Resize and crop the image to 224x224
    size = (224, 224)
    image = image_to_predict.resize(size)

    # Convert image to numpy array and normalize it
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # A TFSMLayer is called like a function, not with .predict()
    # The output is a dictionary, so we get the tensor from its values.
    prediction_output = model_layer(data)
    prediction_tensor = list(prediction_output.values())[0]

    # Get the prediction array
    prediction = prediction_tensor.numpy()[0]

    # Find the index of the highest probability
    index = np.argmax(prediction)
    class_name = labels[index]
    confidence_score = prediction[index]

    return class_name, confidence_score

# --- MAIN APP INTERFACE ---

# 1. Title and How-to-Use Guide
st.title("🔥 Welding Anomaly Detection")
with st.expander("ℹ How to Use This App"):
    st.write("""
        1. *Upload an image* of a welded joint using the file uploader.
        2. The AI model will analyze the image for potential anomalies or defects.
        3. The *Status* and *Confidence Score* will be displayed on the right.
        4. *'Normal'* indicates the weld has passed inspection. *'Anomaly'* suggests a potential defect was found.
    """)

# 2. File Uploader for User's Image
uploaded_file = st.file_uploader("Upload a weld image for inspection...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")

    st.header("Analysis Results")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Your Image")
        st.image(image, use_column_width=True)

    with col2:
        st.subheader("Prediction")
        # Make a prediction
        with st.spinner('Analyzing the image...'):
            # Load the model and labels
            model_layer, labels = load_keras_model()
            class_name, confidence_score = predict(image, model_layer, labels)

        # Display the richer, color-coded result
        if "anomaly" in class_name.lower(): # More flexible check for the anomaly class
            st.error(f"Status: {class_name}")
            st.write(f"*Confidence:* {confidence_score:.2%}")
            st.warning("*Recommendation:* This weld should be flagged for manual inspection.")
        else:
            st.success(f"Status: {class_name}")
            st.write(f"*Confidence:* {confidence_score:.2%}")
            st.info("*Recommendation:* This weld has passed the automated inspection.")
else:
    # 3. Example Cases when no file is uploaded
    st.header("Example Cases")
    st.write("No image uploaded yet. Check out these examples of normal and anomalous welds:")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Normal Weld")
        # IMPORTANT: Replace with the actual path to your normal weld example image
        st.image("normal_weld_example.jpg", caption="A weld that would pass inspection.")
    with col2:
        st.subheader("Anomalous Weld")
        # IMPORTANT: Replace with the actual path to your anomalous weld example image
        st.image("anomalous_weld_example.jpg", caption="A weld with a potential defect.")
