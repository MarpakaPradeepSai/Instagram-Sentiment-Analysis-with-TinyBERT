import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

# Streamlit app
st.title("Sentiment Analysis with TinyBERT")

# Load the model and tokenizer
@st.cache_resource
def load_model():
    model_path = "TinyBERT_model"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

tokenizer, model = load_model()

# Function to predict sentiment
def predict_sentiment(text):
    if tokenizer is None or model is None:
        return "Model not loaded"
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probs, dim=-1).item()
    
    sentiment_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return sentiment_mapping[predicted_class]

# Text input
user_input = st.text_area("Enter text for sentiment analysis:", "")

if st.button("Analyze Sentiment"):
    if user_input:
        sentiment = predict_sentiment(user_input)
        st.write(f"Predicted sentiment: {sentiment}")
    else:
        st.write("Please enter some text to analyze.")

# Add some information about the model
st.sidebar.header("About")
st.sidebar.info("This app uses a fine-tuned TinyBERT model to predict the sentiment of text as Negative, Neutral, or Positive.")

# Debug information
st.sidebar.header("Debug Info")
st.sidebar.write(f"Model path exists: {os.path.exists('TinyBERT_model')}")
st.sidebar.write(f"Files in model directory: {os.listdir('TinyBERT_model') if os.path.exists('TinyBERT_model') else 'N/A'}")
