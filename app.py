import streamlit as st
from PIL import Image
import numpy as np
import easyocr
from transformers import pipeline

# Title
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detector")
st.write("Upload an image with news text, and we will detect if it's real or fake!")

# Load EasyOCR Reader (English only, CPU)
@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr_reader()

# Load Transformer Classifier
@st.cache_resource
def load_classifier():
    return pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news")

classifier = load_classifier()

# Function: Extract text from image
def extract_text_from_image(image_file):
    try:
        # Load and convert image to numpy array
        image = Image.open(image_file).convert("RGB")
        image_np = np.array(image)

        # OCR text extraction
        result = reader.readtext(image_np, detail=0)
        return " ".join(result)

    except Exception as e:
        return f"❌ OCR Error: {str(e)}"

# Streamlit file uploader
image_file = st.file_uploader("📤 Upload an image (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"])

# If file is uploaded
if image_file:
    st.image(image_file, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Extract Text & Detect Fake News"):
        with st.spinner("🔄 Processing..."):
            # Extract text
            extracted_text = extract_text_from_image(image_file)

            # Handle OCR failure
            if extracted_text.startswith("❌ OCR Error"):
                st.error(extracted_text)
            elif not extracted_text.strip():
                st.warning("⚠ No readable text found in the image.")
            else:
                st.success("✅ Text successfully extracted!")

                # Show extracted text
                st.markdown("### 📝 Extracted Text:")
                st.write(extracted_text)

                # Classify the text
                prediction = classifier(extracted_text)[0]
                label = prediction['label']
                score = prediction['score']

                st.markdown("### 🧠 Fake News Prediction")
                st.markdown(f"*Label:* {label}")
                st.markdown(f"*Confidence:* {score:.2f}")

# Footer
st.markdown("---")
st.caption("Built with Streamlit, EasyOCR, and Transformers")