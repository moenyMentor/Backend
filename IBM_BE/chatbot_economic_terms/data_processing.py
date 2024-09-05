import re
import os
import pandas as pd
import pymupdf4llm
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

### pdf 파일 처리 
# pdf 문서 전처리
def preprocess_financial_text(text):
    # 문서에 필요없는 앞 뒤 삭제
    start = re.search(r'\*\*', text).span()[0]  # 첫 번재 마크다운이 나오는 위치
    end = re.search(r'경제금융용어 700선 집필자', text).span()[0]
    text = text[start:end]

    # 불필요한 문자 제거
    text = text.replace("경제금융용어  700선", "")
    text = text.replace("\n\n연관검색어", " 연관검색어")
    text = re.sub(r'\n\n\n-----\n\n', ' ', text)
    text = text.replace("\n\n", " ")
    text = re.sub(r'[ㄱ-ㅎ]', '', text)  # 초성 제거

    # 나중에 chunk를 \n\n을 기준으로 할 수 있게 용어 앞 뒤 처리 
    text = text.replace("**", "\n\n")
    text = text.replace("\n\n ", " ")

    return text

# txt 파일로 저장
def save_to_txt_file(file_name, content):
    with open(file_name, 'w', encoding='utf-8') as file:
        file.write(content)

# PDF 로드 및 전처리
def load_and_process_financial_data():
    # pdf_filename = os.path.join(os.getcwd(), 'data', '한국은행_경제금융용어.pdf')
    # financial_md_text = pymupdf4llm.to_markdown(pdf_filename)     # markdown 형식으로 pdf 읽기
    # clean_financial_md_text = preprocess_financial_text(financial_md_text)
    
    # # 문자열을 파일로 저장
    # save_to_txt_file('data/한국은행_경제금융용어.txt', clean_financial_md_text)

    # txt 파일 불러오기
    txt_filename = os.path.join(os.getcwd(), 'chatbot_economic_terms', 'data', '한국은행_경제금융용어.txt')
    # txt_filename = os.path.join(os.getcwd(), 'data', '한국은행_경제금융용어.txt')
    loader = TextLoader(txt_filename, encoding='utf-8')
    financial_documents = loader.load()    

    # chunck "/n/n"을 기준으로
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=0
    )

    financial_texts = text_splitter.split_documents(financial_documents)

    return financial_texts
            

### xlsx 파일 처리 
def preprocess_current_text(doc):
    doc.page_content = doc.page_content.replace('\n', ' ')
    return doc

def load_and_process_current_data():
    # xlsx_filename = os.path.join(os.getcwd(), 'data', '기획재정부_시사경제용어사전.xlsx')
    xlsx_filename = os.path.join(os.getcwd(), 'chatbot_economic_terms','data', '기획재정부_시사경제용어사전.xlsx')
    data = pd.read_excel(xlsx_filename)
    # data.to_csv('data/기획재정부_시사경제용어사전.csv', index=False)
    data.to_csv(os.path.join(os.getcwd(), 'chatbot_economic_terms','data', '기획재정부_시사경제용어사전.csv'), index=False)
    # csv_filename = os.path.join(os.getcwd(), 'data', '기획재정부_시사경제용어사전.csv')
    csv_filename = os.path.join(os.getcwd(), 'chatbot_economic_terms', 'data', '기획재정부_시사경제용어사전.csv')
    loader = CSVLoader(csv_filename, encoding='utf-8')
    current_documents = loader.load()

    current_texts = [preprocess_current_text(doc) for doc in current_documents]
    return current_texts    
