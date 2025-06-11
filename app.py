import streamlit as st
from PIL import Image
import numpy as np
import keras # We now need to import keras directly

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

    The model is a Convolutional Neural Network (CNN) trained on a dataset of welding images. It was created with Google's Teachable Machine (using the SavedModel format) and deployed in this Streamlit application.
""")
st.sidebar.success("Project by: Daksh")


# --- MODEL LOADING ---
@st.cache_resource
def load_keras_model():
    """
    Loads the Keras model and labels from the disk.
    This function is designed to load a TensorFlow SavedModel format,
    which is exported from Teachable Machine.
    """
    # --- IMPORTANT ---
    # Make sure 'my_model' is the name of the folder containing your SavedModel files.
    # Make sure 'labels.txt' is the correct name for your labels file.
    labels_path = "labels.txt"
    model_path = "model.savedmodel" # This should be the FOLDER name

    # Load the labels
    with open(labels_path, "r") as f:
        labels = [line.strip() for line in f.readlines()]

    # Load the SavedModel as a special Keras Layer (TFSMLayer)
    # This is the correct method for Keras 3 and TensorFlow 2+
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
    # The image needs to be converted to RGB for models trained on color images
    image = image_to_predict.convert("RGB").resize(size)

    # Convert image to numpy array and normalize it
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # A TFSMLayer is called like a function, not with .predict()
    # The output is a dictionary, so we get the tensor from its values.
    prediction_output = model_layer(data)
    prediction_tensor = list(prediction_output.values())[0]

    # Get the prediction array from the tensor
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
        1. *Upload an image* of a welded joint using the file uploader below.
        2. The AI model will analyze the image for potential anomalies or defects.
        3. The *Status* and *Confidence Score* will be displayed on the right.
        4. *'Normal'* indicates the weld has passed inspection. *'Anomaly'* suggests a potential defect was found.
    """)

# 2. File Uploader for User's Image
uploaded_file = st.file_uploader("Upload a weld image for inspection...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file)

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
            # Run prediction
            class_name, confidence_score = predict(image, model_layer, labels)

        # Display the richer, color-coded result
        # Check if 'anomaly' or 'defective' is in the class name for flexibility
        if any(keyword in class_name.lower() for keyword in ["anomaly", "defective"]):
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
