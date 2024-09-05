from langchain.schema import Document
from langchain.vectorstores import Chroma
from chatbot_finance_guide.model import load_embedding_model

def load_text_as_documents(file_path):
    file_path = file_path
    documents = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, 1):
            line = line.strip()  # 앞뒤 공백 제거
            if line:  # 빈 줄 무시
                doc = Document(
                    page_content=line,
                    metadata={"source": file_path, "line_number": line_number}
                )
                documents.append(doc)

    return documents

texts = (load_text_as_documents("./chatbot_finance_guide/data/saving_list.txt")
         + load_text_as_documents("./chatbot_finance_guide/data/deposit_list.txt")
         + load_text_as_documents("./chatbot_finance_guide/data/CMA_list.txt")
         + load_text_as_documents("./chatbot_finance_guide/data/fund_list.txt")
         + load_text_as_documents("./chatbot_finance_guide/data/ELS_list.txt") 
         + load_text_as_documents("./chatbot_finance_guide/data/채권형펀드 리스트.txt"))


DB_PATH = "./chatbot_finance_guide/chroma_db"

def create_vector_db(texts, embeddings):
    db = Chroma.from_documents(
        documents=texts, 
        embedding=embeddings, 
        persist_directory=DB_PATH, 
        collection_name="investment_products_db"
    )
    return db

def load_vector_db():
    embeddings = load_embedding_model()
    return Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings,
        collection_name="investment_products_db",
    )