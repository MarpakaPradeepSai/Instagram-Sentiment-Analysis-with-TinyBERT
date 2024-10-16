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
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 10px 24px;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📊 Sentiment Analysis with TinyBERT")
st.markdown(
    """
    <div style="background-color:#f4f4f8; padding: 10px; border-radius: 10px; margin-bottom: 20px;">
        <h2 style="color: #4CAF50;">Analyze the sentiment of your text</h2>
        <p>Enter a piece of text and let our TinyBERT model determine if it's Positive, Neutral, or Negative.</p>
    </div>
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
            <div style="background-color:#e7f5e9; padding: 10px; border-radius: 5px;">
                <h3>Sentiment: {sentiment_label}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.write("Please enter text to analyze.")
