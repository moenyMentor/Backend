from langchain.vectorstores import Chroma
from chatbot_economic_terms.data_processing import load_and_process_financial_data
from chatbot_economic_terms.data_processing import load_and_process_current_data
from chatbot_economic_terms.model import load_embedding_model

# DB_PATH = "./chroma_db"
DB_PATH = "./chatbot_economic_terms/chroma_db"

financial_texts = load_and_process_financial_data()
current_texts = load_and_process_current_data()

def create_vector_db(financial_texts, current_texts, embeddings):
    db = Chroma.from_documents(
        documents=financial_texts, 
        embedding=embeddings, 
        persist_directory=DB_PATH, 
        collection_name="economic_terms_db"
    )
    db.add_documents(documents=current_texts)
    return db

def load_vector_db():
    embeddings = load_embedding_model()
    return Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings,
        collection_name="economic_terms_db",
    )
