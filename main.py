import os
from dotenv import load_dotenv
import streamlit as st
import fitz 
from langchain.vectorstores import FAISS
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate

# Explicitly load environment variables from .env file
load_dotenv()

# Access OpenAI API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

# Check if OpenAI API key is available
if openai_api_key is None:
    raise ValueError("OpenAI API key not found. Please make sure it's set in your .env file.")

# Initialize OpenAI and embeddings
llm = OpenAI(temperature=0.9, openai_api_key=openai_api_key)
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Define the Document class
class Document:
    def __init__(self, page_content):
        self.page_content = page_content
        self.metadata = ""  

# Define the main function
def main():
    st.title("Question Maker")

    if 'show_button' not in st.session_state:
        st.session_state.show_button = True

    if 'question' not in st.session_state:
        st.session_state.question = None

    uploaded_file = st.file_uploader("Drag and drop a PDF file here", type="pdf")

    if uploaded_file is not None:
        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
        pdf_data = uploaded_file.read()
        text_content = extract_text_from_pdf(pdf_data)
        documents = [Document(page_content) for page_content in text_content]
        docsearch = FAISS.from_documents(documents, embeddings)
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())
        
        if st.session_state.show_button:
            if st.button("Generate New Question"):
                st.session_state.question = generate_question(llm, text_content)
                st.write("Generated Question:", st.session_state.question)
                st.session_state.show_button = False

        if not st.session_state.show_button:
            user_answer = st.text_input("Answer the question:", "")
            if st.button("Submit"):
                if not st.session_state.question:
                    st.write("Please generate a question first.")
                else:
                    result = check_answer(llm, qa, st.session_state.question, user_answer, text_content)
                    st.write("Your answer is:", result)
                    st.session_state.show_button = True

# Function to extract text from PDF
def extract_text_from_pdf(pdf_data):
    doc = fitz.open(stream=pdf_data, filetype="pdf")
    text_content = [page.get_text() for page in doc]
    return text_content

# Function to generate a question
def generate_question(llm, text_content):
    prompt_template = PromptTemplate(
        input_variables=["text"],
        template="What is a question related to the given text: '{text}' ENSURE TO NEVER REPEAT QUESTIONS, AND ONLY GIVE ONE QUESTION!!! Also, ENSURE THE QUESTIONS YOU ASK CAN BE FULLY ANSWERED BASED ON THE GIVEN TEXT!"
    )
    prompt = prompt_template.format(text=text_content)
    return llm(prompt)

# Function to check the answer
def check_answer(llm, qa, question, user_answer, text_content):
    prompt_template = PromptTemplate(
        input_variables=["text"],
        template="Is the following statement true or false based on the given text: '{text}'? The question: '{question}' The user's answer: {user_answer}. Answer ONLY with 'Yes' or 'No'. Be lenient with your answer, if the user is close, just say yes"
    )
    prompt = prompt_template.format(text=text_content, question=question, user_answer=user_answer)
    response = llm(prompt)
    if response.lower() == "yes":
        return "correct"
    else:
        correct_answer = qa.run(question)
        st.write("The correct answer is:", correct_answer)
        return "wrong"

# Run the main function
if __name__ == "__main__":
    main()
