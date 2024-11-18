import os
import openai
import numpy as np
import pandas as pd
import json
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from openai.embeddings_utils import get_embedding
import faiss
import streamlit as st
import warnings
from streamlit_option_menu import option_menu
from streamlit_extras.mention import mention
import requests
from bs4 import BeautifulSoup
import PyPDF2

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Custom Soup", page_icon="", layout="wide")
if 'wall_of_text' not in st.session_state:
    st.session_state.wall_of_text = ""
if 'domain' not in st.session_state:
    st.session_state.domain = ""
if 'chunks' not in st.session_state:
    st.session_state.chunks = False
if 'index' not in st.session_state:
    st.session_state.index = False

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
    st.session_state["api_key"] = ""
    st.session_state["initial_login_state"] = False

def verify_api_key(api_key):
    try:
        # Set the OpenAI API key
        openai.api_key = api_key
        
        # Make a small test request to verify if the key is valid
        openai.Model.list()
        
        # If the request is successful, return True
        return True
    except Exception as e:
        # If there's an error, the API key is likely invalid
        return False

def log_in():
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2 :
        st.title("Input OpenAI API Key")
    api_key = st.text_input("Input OpenAI API Key here", type="password")
    
    if st.button("Log In"):
        if verify_api_key(api_key):
            st.session_state["logged_in"] = True
            st.session_state["api_key"] = api_key
            st.session_state["initial_login_state"] = True
            st.success("Login successful!")
            
            # Use st.query_params to set the logged_in query param
            st.query_params = {"logged_in": "true"}
            st.rerun()
        else:
            st.error("Invalid credentials. Enter valid API Key.")

def Home():
    st.title('Custom Soup is here!')
    st.write("Welcome to Custom Soup, a unique chatbot solution designed to make customer support smarter and more efficient.")
    st.write("Our chatbot specializes in turning your website content or PDF files into a powerhouse of customer assistance by extracting and embedding key information directly into a language model.")
    st.write("This means your customers get faster, more accurate answers without the need for manual intervention.")    
    st.write("## Features:")
    st.write("- Data Extraction Made Easy: Simply upload a PDF file or provide a website link, and Custom Soup will process the data to build a knowledge base tailored to your needs.")
    st.write("- LLM-Powered Responses: With advanced language model integration, our chatbot delivers precise and context-aware support for customer inquiries.")
    st.write("- Seamless Setup: No complex configurations—just plug in your data source, and you're ready to go.")
    st.write("- Enhanced Customer Experience: Provide faster resolutions and consistent support for a better overall interaction.")
    st.write("With Custom Soup, you’ll serve your customers a personalized and satisfying support experience every time! 🍜")

def extract_text_from_url(url):
    """Fetches and extracts text from a given URL."""
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup.get_text(strip=True)  # Use strip=True to clean up whitespace

def find_and_extract_links(main_url):
    """Finds all links on the main page and extracts text from each."""
    response = requests.get(main_url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all anchor tags
    links = soup.find_all('a')

    texts = []  # Initialize a list to store text data

    # Extract href attributes and texts
    for link in links:
        href = link.get('href')
        if href and href.startswith('http'):  # Check if it's a valid URL
            print(f'Extracting from: {href}')
            text = extract_text_from_url(href)
            texts.append(text)  # Append the extracted text to the list

    # Join all texts into a single string
    all_texts = "\n\n".join(texts)  # Use double newlines for better separation
    return all_texts

def extract_text_from_pdf(file): #Extract text from a PDF file-like object.
    text = ""
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        text += page.extract_text() or ""  # Handle cases where extract_text() returns None
    return text

def embeddings():
    chunk_size = 1000
    st.session_state.chunks = [st.session_state.wall_of_text[i:i+chunk_size] for i in range(0, len(st.session_state.wall_of_text), chunk_size)]
    embeddings = [get_embedding(chunk, engine='text-embedding-3-small') for chunk in st.session_state.chunks]
    embedding_dim = len(embeddings[0])
    embeddings_np = np.array(embeddings).astype('float32')
    st.session_state.index = faiss.IndexFlatL2(embedding_dim)
    st.session_state.index.add(embeddings_np) 

def CurrentDomain():
    st.title('Here you can find the current domain expertise of this chatbot')
    st.write(f'The current domain embedded into this chatbos is about {st.session_state.domain}.')

def Domain_Expansion():
    st.title('Welcome to Domain Expansion')
    st.write("Upload your link or PDF file to update my domain here")
    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        domain = st.text_input("What will the new domain?", placeholder="Input Domain Name")
        main_page_url = st.text_input("Use a documentation link for the new domain", placeholder="Input website link here")
        st.write("or")
        uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"], accept_multiple_files=False)
        submit_button = st.button("Submit")

    if submit_button:
        with st.spinner(f"Domain Expansion, uploading {domain} knowledge"):
            all_extracted_texts = ""
            if main_page_url != "":
                all_extracted_texts = find_and_extract_links(main_page_url)
            text = ""
            if uploaded_file is not None:
                text = extract_text_from_pdf(uploaded_file)
            
            # Update wall_of_text in session state
            if all_extracted_texts and text:
                st.session_state.wall_of_text = f"{all_extracted_texts}  {text}"
            elif all_extracted_texts:
                st.session_state.wall_of_text = all_extracted_texts
            elif text:
                st.session_state.wall_of_text = text
            
            st.session_state.domain = domain
            embeddings()
            st.write(f"Your domain on {st.session_state.domain} has been uploaded!")
            
            # Add these lines to redirect to Chat page
            st.session_state['current_page'] = "Chat"
            st.rerun()

def Chat():
    st.title(f'Chat with your enhanced chatbot with reinforced {st.session_state.domain} knowledge!')
    user_message = st.text_input(f'Ask your chatbot anything {st.session_state.domain} related and it will help you with the best of its ability.')                
    submit_chat_button = st.button("Submit Chat")
    if submit_chat_button:
        with st.spinner(f"Domain Expansion, checking my {st.session_state.domain} knowledge!"):
            system_prompt = f"""
                Role:

                You are a specialized Customer Service and Data Analyst Chatbot with {st.session_state.domain}. Your expertise lies in processing and utilizing data uploaded by users (via links or PDF files) to provide precise, domain-specific answers. You do not handle general knowledge inquiries but instead guide users to an alternative chatbot, like ChatGPT, for such queries. Your primary goal is to enhance customer service and provide insights directly tied to the uploaded content.
                Instructions:

                    Data Integration: Accept and process user-uploaded content (links or PDFs), extracting and structuring relevant information into your knowledge base for immediate use.
                    Focused Assistance: Use only the uploaded data to answer user questions. Remain within the scope of the provided domain.
                    Professionalism: Communicate in a clear, concise, and professional tone, ensuring user satisfaction and trust.
                    General Knowledge Requests: If a user asks a question unrelated to the uploaded data, politely decline and suggest they consult ChatGPT for general inquiries.
                    Session-Based Knowledge: Retain knowledge only for the current session. Do not store or share data beyond this context to ensure privacy and compliance.

                Context:

                You operate in situations where users need:

                    A reliable tool to analyze and leverage specific data sources, such as reports, documents, or website content.
                    A focused chatbot that provides answers tied directly to the uploaded information.
                    Assistance for customer support or data analysis tasks that benefit from a clear, context-aware approach.

                You cater to professionals, businesses, or researchers who need precise, domain-specific insights. General-purpose answers fall outside your scope, and such questions are better suited for ChatGPT.
                Constraints:

                    Domain Specificity: Only use knowledge from the provided links or PDFs to craft responses. Avoid speculative or unrelated answers.
                    No General Knowledge: Do not answer questions outside the domain of the uploaded content. Redirect users to ChatGPT or other relevant resources.
                    Privacy Compliance: Ensure that uploaded data is only used for the current session and is not stored, shared, or retained after the session ends.
                    Accuracy and Clarity: Responses should be factually accurate, concise, and directly relevant to the uploaded data. Avoid vague or overly broad answers.
                    Respect Boundaries: If the uploaded content is unclear or incomplete, inform the user and ask for clarification or additional input to ensure accurate assistance.

                Examples:

                Example 1: Domain-Specific Inquiry
                User: "What insights can you provide from the uploaded annual report about customer demographics?"
                Chatbot: "The annual report indicates that 60% of your customers are between the ages of 25-34, with a significant concentration in urban areas. This is outlined in the demographic section on page 12."

                Example 2: Request Outside Scope
                User: "What is the weather like in Paris today?"
                Chatbot: "I specialize in answering questions based on the data you’ve uploaded. For information about current weather, I recommend asking ChatGPT or a weather-specific resource."

                Example 3: Mixed Domain-Specific Inquiry
                User: "How does the uploaded market analysis compare to global industry growth trends?"
                Chatbot: "The uploaded market analysis shows a 7% annual growth rate for your industry, which aligns with the global industry average of 6-8% as mentioned on page 7. For further details, consult the section titled 'Global Benchmarks' in the uploaded file."

                Example 4: Incomplete Data Request
                User: "What can you say about product performance trends?"
                Chatbot: "The uploaded data doesn’t seem to include specific details about product performance trends. If you have additional files or information, please upload them so I can assist further."
            
            """
            query_embedding = get_embedding(user_message, engine='text-embedding-3-small')
            query_embedding_np = np.array([query_embedding]).astype('float32')
            _, indices = st.session_state.index.search(query_embedding_np, 20)
            retrieved_docs = [st.session_state.chunks[i] for i in indices[0]]
            context = ' '.join(retrieved_docs)
            struct = [{"role": "system", "content": system_prompt}]
            structured_prompt = f"Context:\n{context}\n\nQuery:\n{user_message}\n\nResponse:"
            chat =  openai.ChatCompletion.create(model = "gpt-4o-mini", messages = struct + [{"role": "user", "content" : structured_prompt}], temperature=0.5, max_tokens=1500, top_p=1, frequency_penalty=0, presence_penalty=0)
            struct.append({"role": "user", "content": user_message})
            response = chat.choices[0].message.content
            struct.append({"role": "assistant", "content": response})
            st.success("Here's what I have...")
            st.write(response)
        

def main_page():
    with st.sidebar :
        st.image("custom_soup.png", use_column_width=True)
        
        with st.container() :
            l, m, r = st.columns((1, 3, 1))
            with l : st.empty()
            with m : st.empty()
            with r : st.empty()
    
        options = option_menu(
            "Dashboard", 
            ["Home", "Add Domain", "Chat", f"{st.session_state.domain} Domain"],
            icons = ['house', 'globe', 'robot', 'pin'],
            menu_icon = "book", 
            default_index = 0,
            styles = {
                "icon" : {"color" : "#dec960", "font-size" : "20px"},
                "nav-link" : {"font-size" : "17px", "text-align" : "left", "margin" : "5px", "--hover-color" : "#262730"},
                "nav-link-selected" : {"background-color" : "#262730"}
            })
    
    # Add this at the beginning of routing logic
    if 'current_page' in st.session_state:
        options = st.session_state['current_page']
        del st.session_state['current_page']  # Clear the redirect after using it

    if 'messages' not in st.session_state :
        st.session_state.messages = []

    if st.session_state.get("initial_login_state"):
        Home()
        st.session_state["initial_login_state"] = False  # Reset after redirect
        
    if 'chat_session' not in st.session_state :
        st.session_state.chat_session = None
        
    elif options == "Home" :
        Home()

    elif options == "Add Domain" :
        Domain_Expansion()

    elif options == "Chat" :
        Chat()

    elif options == f"{st.session_state.domain} Domain" :
        CurrentDomain()

query_params = st.query_params  # Use st.query_params for retrieval
if query_params.get("logged_in") == ["true"] or st.session_state["logged_in"]:
    main_page()
else:
    log_in()