import json
import sys
from pathlib import Path

import joblib
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

# Streamlit executes this file from the app/ directory. Add the project root
# explicitly so the shared preprocessing.py module can be imported reliably.
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from preprocessing import predict_digit

MODEL_PATH = BASE_DIR / "artifacts" / "final_model.joblib"
METADATA_PATH = BASE_DIR / "artifacts" / "metadata.json"

st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="✏️",
    layout="centered"
)

st.title("MNIST Handwritten Digit Classifier")
st.write(
    "Ladda upp ett foto av en handskriven siffra. "
    "Bilden bearbetas till samma 28×28-format som MNIST-modellen använder."
)

if not MODEL_PATH.exists():
    st.error(
        "Modellen saknas. Kör notebooken först så att "
        "artifacts/final_model.joblib skapas."
    )
    st.stop()


@st.cache_resource
def load_model(path, mtime):
    return joblib.load(path)


model = load_model(MODEL_PATH, MODEL_PATH.stat().st_mtime)

metadata = {}
if METADATA_PATH.exists():
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

uploaded_file = st.file_uploader("Välj en bild", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    original = ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")
    st.image(original, caption="Originalbild", width=600)

    try:
        prediction, probabilities, (gray, mask, canvas) = predict_digit(
            model, np.asarray(original), return_debug=True
        )
        top3 = np.argsort(probabilities)[::-1][:3]

        st.subheader("Preprocessing")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(gray, caption="Grayscale", width=200)
        with col2:
            st.image(mask, caption="Detected ink mask", width=200)
        with col3:
            st.image(canvas, caption="Final 28×28", width=200)

        st.subheader("Prediction")
        st.metric("Predicted digit", prediction)

        st.write("Top 3 predictions")
        for cls in top3:
            st.write(f"**{int(cls)}** — {probabilities[cls] * 100:.1f}%")

        st.progress(
            float(probabilities[prediction]),
            text=f"Confidence: {probabilities[prediction] * 100:.1f}%"
        )

        if probabilities[prediction] < 0.35:
            st.warning(
                "Låg säkerhet. Kontrollera bläckmasken ovan: om den innehåller "
                "papperstextur eller saknar delar av siffran, ta en ny bild med "
                "bättre kontrast."
            )

        with st.expander("Model information"):
            st.write(f"Model: {metadata.get('model', 'Extra Trees')}")
            st.write(f"Estimators: {metadata.get('n_estimators', '—')}")
            st.write(f"Input features: {metadata.get('features', 784)}")
            if "augmentation" in metadata:
                st.write(f"Augmentation: {metadata['augmentation']}")
            if "test_accuracy" in metadata:
                st.write(f"Test accuracy: {metadata['test_accuracy']:.4f}")
            st.write(
                "Prediktionen medelvärdesbildas över 15 varianter av bilden "
                "(tre strecktjocklekar × fem förskjutningar)."
            )

    except Exception as exc:
        st.error(f"Could not process the image: {exc}")
