import os
from dotenv import load_dotenv
import requests
import chromadb
import json
import os
import re   # 정규표현식 활용
from dotenv import load_dotenv
import pandas as pd   # 데이터 처리
from langchain_community.document_loaders import TextLoader   # txt 파일 load
from langchain_text_splitters import RecursiveCharacterTextSplitter

#%%
# .env 파일에서 API 키와 URL 가져오기
load_dotenv()
api_key = os.getenv("API_KEY")
api_url = os.getenv("API_URL")

def fetch_data():
    # 요청 헤더 설정
    api_url=f"http://finlife.fss.or.kr/finlifeapi/savingProductsSearch.json?auth={api_key}&topFinGrpNo=020000&pageNo=1"
    print(api_url)
    try:
        # API 요청 보내기
        response = requests.get(api_url)
        response.raise_for_status() 

        # 응답 상태 코드 출력
        print(f"Status Code: {response.status_code}")

        # 응답이 JSON인 경우 출력
        if response.status_code == 200:
            print("Request successful!")
            data = response.json()  # JSON 응답 파싱
            #print("Response JSON:", json.dumps(data, indent=2))  # 확인용
        else:
            print(f"Request failed with status code {response.status_code}")
            print("Response content:", response.content)
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
    
    #불러온 데이터를 data.txt로 저장
    data_str = str(data)  # 필요한 형식으로 변환
    os.makedirs("./data", exist_ok=True)
    file_path = 'data/적금 리스트.txt'
    with open(file_path, 'w',encoding='utf-8') as file:
        file.write(data_str)

    print(f"api에서 호출한 데이터가 '{file_path}'에 저장되었습니다.")

if __name__ == "__main__":
    fetch_data()


#%%
from langchain_community.document_loaders import TextLoader   # txt 파일 load
from tqdm import tqdm
import pandas as pd

#%%
loader = TextLoader("./data/적금 리스트.txt", encoding='utf-8')
api_data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

split_data = text_splitter.split_documents(api_data)

client = chromadb.PersistentClient()
answers = client.create_collection(
    name="answers"
)
#모델 불러오기
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

#데이터 삽입
df = loader.load("./data/적금 리스트.txt")
ids = []
metadatas = []
embeddings = []

for row in tqdm(df.iterrows()):
    index = row[0]
    query = row[1].user
    answer = row[1].answer
    
    metadata = {
        "query": query,
        "answer": answer
    }
    
    embedding = model.encode(query, normalize_embeddings=True)
    
    ids.append(str(index))
    metadatas.append(metadata)
    embeddings.append(embedding)
    
chunk_size = 100  # 한 번에 처리할 chunk 크기 설정
total_chunks = len(embeddings) // chunk_size + 1  # 전체 데이터를 chunk 단위로 나눈 횟수
embeddings = [ e.tolist() for e in tqdm(embeddings)]  

for chunk_idx in tqdm(range(total_chunks)):
    start_idx = chunk_idx * chunk_size
    end_idx = (chunk_idx + 1) * chunk_size
    
    # chunk 단위로 데이터 자르기
    chunk_embeddings = embeddings[start_idx:end_idx]
    chunk_ids = ids[start_idx:end_idx]
    chunk_metadatas = metadatas[start_idx:end_idx]
    
    # chunk를 answers에 추가
    answers.add(embeddings=chunk_embeddings, ids=chunk_ids, metadatas=chunk_metadatas)
