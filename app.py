import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load the model and tokenizer
model_path = "TinyBERT_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Function to predict sentiment
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probs, dim=-1).item()
    
    sentiment_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return sentiment_mapping[predicted_class]

# Streamlit app
st.title("Sentiment Analysis with TinyBERT")

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
