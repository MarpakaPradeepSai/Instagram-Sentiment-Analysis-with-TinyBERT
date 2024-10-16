import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load tokenizer and model
model_path = "./TinyBERT_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Function to predict sentiment
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    return probs.detach().numpy()

# Streamlit app
st.title("Sentiment Analysis with TinyBERT")
user_input = st.text_area("Enter text to analyze")

if st.button("Analyze"):
    if user_input:
        sentiment_probs = predict_sentiment(user_input)
        st.write(f"Sentiment probabilities: {sentiment_probs}")
    else:
        st.write("Please enter text to analyze.")
