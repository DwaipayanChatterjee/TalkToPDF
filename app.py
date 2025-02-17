import streamlit as st
import PyPDF2
import pdfplumber
from transformers import pipeline

# Load the pre-trained transformer model for question answering
qa_model = pipeline("question-answering")

# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Streamlit UI
st.title("PDF Question Answering System")

# Step 1: Upload the PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("uploaded_file.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Extract text from PDF
    pdf_text = extract_text_from_pdf("uploaded_file.pdf")
    
    # Display a preview of the extracted text (optional)
    st.subheader("Preview of Extracted Text")
    st.text(pdf_text[:1000])  # Show the first 1000 characters for preview

    # Step 2: Ask a question
    question = st.text_input("Ask a question related to the PDF:")
    
    if question:
        # Get the answer using the question-answering model
        answer = qa_model(question=question, context=pdf_text)
        st.subheader("Answer:")
        st.write(answer['answer'])
