import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
import os
import torch  # Import torch module

# Load the models
summarization_model_name = 'facebook/bart-large-cnn'
tokenizer = AutoTokenizer.from_pretrained(summarization_model_name)
summarization_model = AutoModelForSeq2SeqLM.from_pretrained(summarization_model_name)

qa_model_name = 'deepset/bert-large-uncased-whole-word-masking-squad2'
qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)

# Function to extract text from a PDF file
def extract_text_from_pdf(file):
    doc = fitz.open(file)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to summarize document
def summarize_document(document):
    inputs = tokenizer(document, return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = summarization_model.generate(inputs['input_ids'], max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Function to get answer to question
def get_answer(question, context):
    inputs = qa_tokenizer(question, context, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = qa_model(**inputs)
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores) + 1
    answer = qa_tokenizer.convert_tokens_to_string(qa_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
    return answer

# Streamlit app
st.title("PDF Summarizer and Q&A")
st.write("Upload a PDF file to get a summary and ask questions about the content.")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Extract text from the PDF
    document_text = extract_text_from_pdf("temp.pdf")
    
    # Display the extracted text
    st.write("Extracted Text:")
    st.write(document_text)
    
    if st.button("Summarize"):
        with st.spinner('Summarizing...'):
            summary = summarize_document(document_text)
            st.write("**Summary:**")
            st.write(summary)
    
    question = st.text_input("Ask a question about the document")
    
    if st.button("Get Answer"):
        if question:
            with st.spinner('Generating answer...'):
                answer = get_answer(question, document_text)
                st.write("**Answer:**")
                st.write(answer)
        else:
            st.write("Please enter a question.")

# Remove temporary file after use
if os.path.exists("temp.pdf"):
    os.remove("temp.pdf")
