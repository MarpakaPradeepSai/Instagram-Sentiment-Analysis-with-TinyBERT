import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import requests
import os

# Function to download model files from GitHub
def download_file_from_github(url, local_path):
    response = requests.get(url)
    with open(local_path, 'wb') as f:
        f.write(response.content)

# URLs of your model files on GitHub
repo_url = 'https://github.com/MarpakaPradeepSai/Instagram-Sentiment-Analysis-with-TinyBERT/raw/main/TinyBERT_model'
files = ['config.json', 'model.safetensors', 'special_tokens_map.json', 'tokenizer_config.json', 'training_args.bin', 'vocab.txt']

# Download each file
for file in files:
    download_file_from_github(f"{repo_url}/{file}", f"./TinyBERT_model/{file}")

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('./TinyBERT_model')
model = BertForSequenceClassification.from_pretrained('./TinyBERT_model')

# Function to predict sentiment
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    return probs.detach().numpy()

# Function to map probabilities to sentiment labels
def get_sentiment_label(probs):
    sentiment_mapping = ["Negative", "Neutral", "Positive"]
    max_index = probs.argmax()
    return sentiment_mapping[max_index]

# Streamlit app
st.set_page_config(
    page_title="Sentiment Analysis with TinyBERT",
    page_icon=":mag:",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main {
        background-color: #F0F2F6;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 10px 24px;
        cursor: pointer;
    }
    .center-image {
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📊 Sentiment Analysis with TinyBERT")

st.markdown(
    """
    <img src="https://webcmstavtech.tav.aero/uploads/59f9875dc0e79a3594308ad3/static-pages/main-images/sentiment-analysis_1.jpg" alt="Sentiment Analysis" class="center-image" width="400">
    """,
    unsafe_allow_html=True
)

user_input = st.text_area("Enter text to analyze")

if st.button("Analyze"):
    if user_input:
        sentiment_probs = predict_sentiment(user_input)
        sentiment_label = get_sentiment_label(sentiment_probs[0])  # Get the label for the highest probability
        st.markdown(
            f"""
            <div style="background-color:#e7f5e9; padding: 10px; border-radius: 5px; text-align: center;">
                <h3>Sentiment: {sentiment_label}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.write("Please enter text to analyze.")
