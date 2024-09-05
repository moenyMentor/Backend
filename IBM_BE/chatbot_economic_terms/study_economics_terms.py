import os
import re   # 정규표현식 활용
from dotenv import load_dotenv
import pymupdf4llm    # pdf markdown 형식으로 load
import pandas as pd   # 데이터 처리
from langchain_community.document_loaders import TextLoader   # txt 파일 load
from langchain_community.document_loaders.csv_loader import CSVLoader # csv 파일 load
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate     # ChatPrompt template
from langchain_core.prompts import PromptTemplate         # Rag-Prompt template 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.runnables import RunnableParallel, RunnablePassthrough  # retriever data 병렬 전달
from langchain_core.output_parsers import StrOutputParser # chain output을 str로 받기 

from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from langchain.llms import WatsonxLLM

load_dotenv()
project_id = os.getenv("PROJECT_ID", None)
wml_credentials = {
    "apikey": os.getenv("API_KEY", None),
    "url": 'https://us-south.ml.cloud.ibm.com'
}

# embedding model, LLM model load
# embedding: huggingface의 snunlp/KR-SBERT embedding model 활용
embeddings = HuggingFaceEmbeddings(
    model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS",       # 한국어 특화 임베딩 모델
)

# LLM: watsonx.ai api를 활용하여 "mistralai/mistral-large" 모델 사용
parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY.value,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 500,
    GenParams.STOP_SEQUENCES: ["<|endoftext|>"]
}

model_id =  'mistralai/mistral-large' # 'meta-llama/llama-3-1-70b-instruct' # ModelTypes.LLAMA_2_70B_CHAT.value #
watsonx_mistralai = WatsonxLLM(
    model_id=model_id,
    url=wml_credentials.get("url"),
    apikey=wml_credentials.get("apikey"),
    project_id=project_id,
    params=parameters
)

# 기본 지식 정보(Knowledge base) 구축
# 1. 경제금융용어 데이터 처리
# 데이터 파일 경로 지정
pdf_filename = os.path.join(os.getcwd(), 'data', '한국은행_경제금융용어.pdf')

# pdf파일 markdown형태의 text로 불러오기 
financial_md_text = pymupdf4llm.to_markdown(pdf_filename)     # markdown 형식으로 pdf 읽기

# 불필요한 문자 제거
def preprocessing_text(text):  
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

# 파일에 문자열을 저장하는 함수
def save_to_txt_file(file_name, content):
    with open(file_name, 'w', encoding='utf-8') as file:
        file.write(content)

# 전처리 
clean_financial_md_text = preprocessing_text(financial_md_text)

# 문자열을 파일로 저장
save_to_txt_file('data/한국은행_경제금융용어.txt', clean_financial_md_text)

# 데이터 파일 경로 지정
txt_filename = os.path.join(os.getcwd(), 'data', '한국은행_경제금융용어.txt')
# txt 파일 불러오기
loader = TextLoader(txt_filename, encoding='utf-8')
financial_documents = loader.load()

# chunck "/n/n"을 기준으로
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=0
)

financial_texts = text_splitter.split_documents(financial_documents)

# 2. 시사경제용어사전 데이터 처리
# xlsx 파일 csv 파일로 변환
xlsx_filename = os.path.join(os.getcwd(), 'data', '기획재정부_시사경제용어사전.xlsx')
data = pd.read_excel(xlsx_filename)
data.to_csv('data/기획재정부_시사경제용어사전.csv', index=False)
csv_filename = os.path.join(os.getcwd(), 'data', '기획재정부_시사경제용어사전.csv')
loader = CSVLoader(csv_filename, encoding='utf-8')
current_documents = loader.load()
# 데이터 전처리
def modify_page_content(doc):
    # 줄바꿈 문자를 공백으로 변경
    doc.page_content = doc.page_content.replace('\n', ' ')
    return doc

# 수정된 Document 객체 리스트 생성
current_texts = [modify_page_content(doc) for doc in current_documents]

# Vector DB 생성: ChromaDB 활용
# 경제금융용어 먼저 넣기
DB_PATH = "./chroma_db"

db = Chroma.from_documents(
    documents=financial_texts, embedding=embeddings, 
    persist_directory=DB_PATH, 
    collection_name="economic_terms_db"
)

# 시사경제용어 추가
db.add_documents(documents=current_texts)

# DB 로드
db = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings,
    collection_name="economic_terms_db",
)

# retriever 생성
retriever = db.as_retriever(search_kwargs={'k': 5})

# prompt 작성
prompt = PromptTemplate.from_template(
    """당신은 질문-답변(Question-Answering)을 수행하는 친절한 AI 어시스턴트입니다. 당신의 임무는 주어진 문맥(context) 에서 주어진 질문(question) 에 답하는 것입니다.
검색된 다음 문맥(context) 을 사용하여 질문(question) 에 답하세요. 만약, 주어진 문맥(context) 에서 답을 찾을 수 없다면, 답을 모른다면 `주어진 정보에서 질문에 대한 정보를 찾을 수 없습니다` 라고 답하세요.
한글로 답변해 주세요. 단, 기술적인 용어나 이름은 번역하지 않고 그대로 사용해 주세요.

#Question: 
{question} 

#Context: 
{context} 

#Answer:"""
)

# chain 생성
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | watsonx_mistralai
    | StrOutputParser()
)

# 질의응답 테스트
user_input = f"{user_input}"
rag_chain.invoke(user_input)