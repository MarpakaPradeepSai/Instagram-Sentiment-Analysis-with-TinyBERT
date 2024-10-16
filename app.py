import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load the model and tokenizer
model_path = './TinyBERT_model'  # Path to your model directory
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path, use_safetensors=False)

# Define sentiment mapping
sentiment_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Streamlit app title
st.title("Sentiment Analysis with TinyBERT")

# Input text area for user
user_input = st.text_area("Enter text for sentiment analysis:")

# Button to submit input
if st.button("Analyze"):
    if user_input:
        # Tokenize input
        inputs = tokenizer(user_input, return_tensors='pt', truncation=True, padding=True)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=1)
            sentiment = sentiment_mapping[predictions.item()]
        
        # Display the result
        st.success(f"The predicted sentiment is: **{sentiment}**")
    else:
        st.warning("Please enter some text.")

# Run the app
if __name__ == "__main__":
    st.run()
