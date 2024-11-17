import streamlit as st
import torch
from transformers import DistilBertTokenizer,DistilBertForSequenceClassification
from transformers import pipeline

MODEL_DIR="./"

def load_model_ans_tokenizer():
    model=DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer=DistilBertTokenizer.from_pretrained(MODEL_DIR)
    sentiment_pipeline=pipeline("text-classification",model=model,tokenizer=tokenizer)
    return sentiment_pipeline

st.title("Sentiment Analysis with fine tuned DistilBERT")
st.write("Enter text for sentiment analysis")
input_text=st.text_area("Enter your feedback")
if st.button("Classifying Sentiment"):
    if input_text:
        sentiment_pipeline=load_model_ans_tokenizer()
        result=sentiment_pipeline(input_text)
        st.write(f"Predicted Sentiment:{result[0]['label']}")
        st.write(f"Confidence:{result[0]['score']:.2f}")
    else:
        st.error("Please enter some text")