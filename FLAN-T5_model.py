import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
import PyPDF2
import io
import re
import spacy

def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()

        

    candidate_name = uploaded_file.name

    return candidate_name, text

def extract_candidate_info(candidate_name, resume_text, nlp, required_skills):
    doc = nlp(resume_text)

    keyword_counts = {skill: len(re.findall(skill, resume_text, flags=re.IGNORECASE)) for skill in required_skills}
    keyword_score = sum(keyword_counts.values())

    return candidate_name, keyword_score

def rank_resumes(resumes, nlp, required_skills):
    ranked_candidates = []

    for candidate_name, resume_text in resumes:
        # Extract candidate information
        candidate_name, keyword_score = extract_candidate_info(candidate_name, resume_text, nlp, required_skills)

        # Append candidate information to the ranked list
        ranked_candidates.append((candidate_name, keyword_score, resume_text))

    # Rank the candidates based on keyword scores
    try:
        ranked_candidates = sorted(ranked_candidates, key=lambda x: x[1], reverse=True)
    except Exception as e:
        st.error(f"Error during ranking: {e}")
        st.write("Printed information for debugging:")
        st.write(ranked_candidates)
        st.write(nlp)
        st.write(required_skills)
        raise e

    return ranked_candidates

def get_pdf_texts(pdf_docs):
    texts = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        texts.append(text)
    return texts

def split_text_into_chunks(text, chunk_size=2048):
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

def get_vectorstore(texts):
    embeddings = HuggingFaceInstructEmbeddings(model_name="google/flan-t5-base")
    all_chunks = []

    for text in texts:
        text_chunks = split_text_into_chunks(text)
        all_chunks.extend(text_chunks)

    vectorstore = FAISS.from_texts(texts=all_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 2048})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Resume Sorter", page_icon=":pdf:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # Define nlp here
    nlp = spacy.load("en_core_web_sm")

    with st.sidebar:
        st.subheader("Your documents")
        # User input for skills
        required_skills = st.text_input("Enter the required skills separated by commas:")

        # Drag-and-drop box for PDFs
        pdf_docs = st.file_uploader("Upload resumes (PDFs only):", type=["pdf"], accept_multiple_files=True)

        # Move the "Process" button below the PDF upload box
        if st.button("Process"):
            with st.spinner("Processing"):
                pdf_texts = get_pdf_texts(pdf_docs)
                vectorstore = get_vectorstore(pdf_texts)
                st.session_state.conversation = get_conversation_chain(vectorstore)

        # Move the "Proceed" button to the sidebar
        proceed_button = st.button("Proceed")

    st.header("Resume Sorter :pdf:")
    user_question = st.text_input("What do you want to know about these resumes..?")
    if user_question:
        handle_userinput(user_question)

    # Use st.form to organize other widgets
    with st.form(key='my_form'):
        st.form_submit_button("Proceed")

    if proceed_button and required_skills and pdf_docs:
        resumes = [extract_text_from_pdf(pdf) for pdf in pdf_docs]
        try:
            ranked_candidates = rank_resumes(resumes, nlp, required_skills.split(","))
        except Exception as e:
            st.error(f"Error in ranking resumes: {e}")
            raise e

        st.header("Ranked Resumes")
        for rank, (candidate_name, keyword_score, resume) in enumerate(ranked_candidates, 1):
            st.markdown(f"Rank: {rank} - Candidate Name: {candidate_name} - Keyword Score: {keyword_score}")

if __name__ == '__main__':
    main()
