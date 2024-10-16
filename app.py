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
    sentiment_mapping = ["Negative 😡", "Neutral 😐", "Positive 😊"]
    max_index = probs.argmax()
    return sentiment_mapping[max_index]

# Streamlit app
st.set_page_config(
    page_title="Sentiment Analysis with TinyBERT",
    page_icon=":mag:",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom CSS to prevent button text color change
st.markdown(
    """
    <style>
    .stButton > button {
        color: black; /* Change this to your desired color */
    }
    .stButton > button:hover {
        color: black; /* Keep the color the same on hover */
    }
    .stButton > button:active {
        color: black; /* Keep the color the same when clicked */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    📊 Instagram Sentiment Analysis with TinyBERT
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
            Sentiment: {sentiment_label}
            """,
            unsafe_allow_html=True
        )
    else:
        st.write("Please enter text to analyze.")
