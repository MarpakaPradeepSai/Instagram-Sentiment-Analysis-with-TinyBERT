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

# Download each file if not already downloaded
if not os.path.exists('./TinyBERT_model'):
    os.makedirs('./TinyBERT_model')
for file in files:
    local_file_path = f"./TinyBERT_model/{file}"
    if not os.path.exists(local_file_path):
        download_file_from_github(f"{repo_url}/{file}", local_file_path)

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

# Streamlit app layout
st.set_page_config(page_title="Sentiment Analysis with TinyBERT", page_icon="💬", layout="wide")
st.title("Sentiment Analysis with TinyBERT")

st.markdown("""
<style>
    .main {
        background-color: #f0f2f5;
        padding: 2rem;
        border-radius: 8px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    }
    h1 {
        color: #4a4a4a;
    }
    .text-input {
        background-color: #ffffff;
        border: 1px solid #d1d1d1;
        border-radius: 4px;
        padding: 10px;
        font-size: 16px;
        width: 100%;
        height: 200px;
    }
    .button {
        background-color: #007bff;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 16px;
    }
    .button:hover {
        background-color: #0056b3;
    }
</style>
""", unsafe_allow_html=True)

user_input = st.text_area("Enter text to analyze:", height=200, key="user_input", help="Type or paste your text here.")

if st.button("Analyze", key="analyze_button", help="Click to analyze sentiment"):
    if user_input:
        sentiment_probs = predict_sentiment(user_input)
        sentiment_label = get_sentiment_label(sentiment_probs[0])
        st.success(f"Sentiment: **{sentiment_label}**")
    else:
        st.error("Please enter text to analyze.")

st.markdown("""
---
### About
This application uses TinyBERT for sentiment analysis. Enter any text to see its sentiment (Negative, Neutral, Positive).
""")
