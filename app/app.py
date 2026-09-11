import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📱",
    layout="centered"
)


# --------------------------------------------------
# Load Model and Tokenizer
# --------------------------------------------------

MODEL_PATH = "models/spam_distilbert_final"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    )

    model.eval()

    return tokenizer, model


tokenizer, model = load_model()


# --------------------------------------------------
# Prediction Function
# --------------------------------------------------

def predict_spam(message):
    inputs = tokenizer(
        message,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=-1)

    predicted_class = torch.argmax(probabilities, dim=-1).item()

    confidence = probabilities[0][predicted_class].item()

    return predicted_class, confidence


# --------------------------------------------------
# User Interface
# --------------------------------------------------

st.title("📱 SMS Spam Detector")

st.write(
    "Enter an SMS message below and the AI model will "
    "predict whether it is **HAM (legitimate)** or **SPAM**."
)

message = st.text_area(
    "Enter your SMS message:",
    placeholder="Example: Congratulations! You have won a free prize..."
)


# --------------------------------------------------
# Prediction Button
# --------------------------------------------------

if st.button("🔍 Check Message"):

    if not message.strip():

        st.warning("Please enter a message before checking.")

    else:

        prediction, confidence = predict_spam(message)

        confidence_percentage = confidence * 100

        if prediction == 1:

            st.error("🚨 SPAM")

            st.write(
                f"Confidence: **{confidence_percentage:.2f}%**"
            )

        else:

            st.success("✅ HAM (Legitimate)")

            st.write(
                f"Confidence: **{confidence_percentage:.2f}%**"
            )


# --------------------------------------------------
# Information
# --------------------------------------------------

st.divider()

st.caption(
    "Model: DistilBERT fine-tuned for SMS spam detection."
)