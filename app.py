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

# Function to map probabilities to sentiment labels and emojis
def get_sentiment_label(probs):
    sentiment_mapping = ["Negative 😡", "Neutral 😐", "Positive 😊"]
    max_index = probs.argmax()
    return sentiment_mapping[max_index]

# Function to get background color based on sentiment
def get_background_color(label):
    if "Positive" in label:
        return "#C3E6CB"  # Softer green
    elif "Neutral" in label:
        return "#FFE8A1"  # Softer yellow
    else:
        return "#F5C6CB"  # Softer red

# Streamlit app
st.set_page_config(
    page_title="Sentiment Analysis with TinyBERT",
    page_icon=":mag:",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom CSS to keep button text white
st.markdown(
    """
    <style>
    .stButton > button {
        color: white;
    }
    .stButton > button:focus {
        color: white;
    }
    .stButton > button:hover {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h1 style="text-align:center;">Instagram Sentiment Analysis with TinyBERT</h1>
    """,
    unsafe_allow_html=True
)

user_input = st.text_area("Enter text to analyze")

if st.button("Analyze"):
    if user_input:
        sentiment_probs = predict_sentiment(user_input)
        sentiment_label = get_sentiment_label(sentiment_probs[0])  # Get the label for the highest probability
        background_color = get_background_color(sentiment_label)  # Get the background color for the sentiment
        st.markdown(
            f"""
            <div style="background-color:{background_color};padding:10px;border-radius:10px">
            <h2 style="text-align:center;">Sentiment: {sentiment_label}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.write("Please enter text to analyze.")
