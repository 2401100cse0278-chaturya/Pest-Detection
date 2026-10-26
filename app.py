import streamlit as st
from ultralytics import YOLO
from PIL import Image
from pathlib import Path

st.set_page_config(page_title="Crop Pest Detection", page_icon="🌱")

st.title("🌱 Crop Pest Detection")
st.write("Upload an image to detect crop pests.")

# Find trained model files
project_dir = Path(__file__).resolve().parent
weights_dir = project_dir / "runs"

model_files = list(weights_dir.rglob("best.pt"))

if not model_files:
    model_files = list(weights_dir.rglob("last.pt"))

if not model_files:
    st.error("Trained model not found. Please check your runs folder.")
    st.stop()

model_path = model_files[0]

@st.cache_resource
def load_model(path):
    return YOLO(path)

model = load_model(str(model_path))

st.success(f"Model loaded: {model_path.name}")

uploaded_file = st.file_uploader(
    "Upload a pest image",
    type=["jpg", "jpeg", "png"]
)

confidence = st.slider(
    "Confidence threshold",
    0.1, 0.9, 0.25, 0.05
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Detect Pest"):
        with st.spinner("Detecting pests..."):
            result = model.predict(
                source=image,
                conf=confidence,
                verbose=False
            )[0]

            output_image = result.plot()[:, :, ::-1]

        st.subheader("Detection Result")
        st.image(output_image, use_container_width=True)

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                class_id = int(box.cls[0])
                label = result.names[class_id]
                score = float(box.conf[0])
                st.write(f"**{label}** — Confidence: {score:.1%}")
        else:
            st.warning("No pests detected in this image.")