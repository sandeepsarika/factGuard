from PIL import Image
import pytesseract
import streamlit as st
import pytesseract
import numpy as np
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
import tweepy

# Load fake news classifier model
from transformers import pipeline
import streamlit as st

st.title("Fake News Detector (Text-based)")
text = st.text_area("Enter news text here:")

if st.button("Analyze"):
    classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english", device=-1)
    result = classifier(text)[0]
    st.write(f"Label: {result['label']}")
    st.write(f"Confidence: {result['score']:.2f}")
# Set your Twitter/X Bearer Token here (optional for now)
TWITTER_BEARER_TOKEN = "YOUR_TWITTER_BEARER_TOKEN"
client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)

# Tesseract path for Windows (uncomment and update if needed)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --- FUNCTION: Detect fake news using model ---
def detect_fake_news(text):
    result = classifier(text)[0]
    return f"{result['label']} (Confidence: {result['score']:.2f})"

# --- FUNCTION: Extract text from uploaded image ---
import streamlit as st
from PIL import Image
import numpy as np
import easyocr

# Load EasyOCR reader once
reader = easyocr.Reader(['en'], gpu=False)

def extract_text_from_image(image_file):
    try:
        # Convert uploaded file to image, then to array
        image = Image.open(image_file).convert("RGB")
        image_np = np.array(image)

        # Perform OCR
        result = reader.readtext(image_np, detail=0)
        return " ".join(result)
    except Exception as e:
        return f"Error extracting text: {str(e)}"

# Streamlit UI
st.title("📰 Fake News Detection from Image")

# Upload image
image_file = st.file_uploader("Upload an image with text", type=["png", "jpg", "jpeg"])

if image_file:
    st.image(image_file, caption="Uploaded Image", use_column_width=True)

    if st.button("Extract & Analyze"):
        extracted_text = extract_text_from_image(image_file)

        if "Error" in extracted_text:
            st.error(extracted_text)
        else:
            st.markdown("### 📜 Extracted Text:")
            st.write(extracted_text)

            # Add your Hugging Face pipeline call here
            # Example:
            # prediction = classifier(extracted_text)[0]
            # st.markdown(f"### 🧠 Prediction: {prediction['label']}")
            # st.markdown(f"*Confidence*: {prediction['score']:.2f}")

# --- FUNCTION: Search related news articles ---
def search_news_articles(query):
    url = f"https://www.google.com/search?q={query}+site:cnn.com+OR+site:bbc.com+OR+site:reuters.com"
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    headlines = [h.get_text() for h in soup.find_all('div', class_='BNeawe vvjwJb AP7Wnd')[:5]]
    return headlines or ["No news articles found."]

# --- FUNCTION: Search related tweets using Twitter API ---
def search_x_posts(query):
    try:
        tweets = client.search_recent_tweets(query=query, max_results=5)
        return [tweet.text for tweet in tweets.data] if tweets.data else ["No recent tweets found."]
    except:
        return ["Twitter API error or missing token."]

# --- STREAMLIT USER INTERFACE ---
st.title("🕵️ Fake News Detector")
st.markdown("Upload news text or image and detect if it's fake, with fact-checking via web and X (Twitter).")

input_mode = st.radio("Choose Input Type:", ("Text", "Image"))

# ---- TEXT INPUT MODE ----
if input_mode == "Text":
    text_input = st.text_area("Enter the news text here:")
    if st.button("Analyze Text"):
        if text_input.strip():
            st.subheader("🔍 Fake News Detection Result")
            st.info(detect_fake_news(text_input))

            st.subheader("📰 Top News Articles")
            for i, news in enumerate(search_news_articles(text_input), 1):
                st.markdown(f"{i}. {news}")

            st.subheader("🐦 Tweets on This Topic")
            for i, tweet in enumerate(search_x_posts(text_input), 1):
                st.markdown(f"{i}. {tweet}")
        else:
            st.warning("Please enter some text to analyze.")

# ---- IMAGE INPUT MODE ----
else:
    image_file = st.file_uploader("Upload an image containing text (e.g., news screenshot):", type=["jpg", "jpeg", "png"])
    if st.button("Analyze Image") and image_file:
        extracted_text = extract_text_from_image(image_file)
        st.subheader("📝 Extracted Text from Image")
        st.code(extracted_text)

        st.subheader("🔍 Fake News Detection Result")
        st.info(detect_fake_news(extracted_text))

        st.subheader("📰 Top News Articles")
        for i, news in enumerate(search_news_articles(extracted_text), 1):
            st.markdown(f"{i}. {news}")

        st.subheader("🐦 Tweets on This Topic")
        for i, tweet in enumerate(search_x_posts(extracted_text), 1):
            st.markdown(f"{i}. {tweet}")
