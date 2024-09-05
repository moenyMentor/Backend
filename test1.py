'''chromadb는 json형식으로 임베딩을 할때 metadata가 필수지만 적금리스트에는 metadata가 없어서?
정상작동 하지 않을 수도 있음?'''


import os
from dotenv import load_dotenv
import requests
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm

# .env 파일에서 API 키와 URL 가져오기
load_dotenv()
api_key = os.getenv("API_KEY")
api_url = os.getenv("API_URL")
company_code = ["020000", "030300"]  # 권역코드
page_number = [1, 3]  # 권역코드별 maxpage
page_counter = 0

# API에서 데이터를 가져오는 함수
def fetch_data(Company_Code, Page_Number):
    api_url = f"http://finlife.fss.or.kr/finlifeapi/savingProductsSearch.json?auth={api_key}&topFinGrpNo={Company_Code}&pageNo={Page_Number}"
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        if response.status_code == 200:
            data = response.json()
        else:
            print(f"Request failed with status code {response.status_code}")
            data = {}
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        data = {}
    # 데이터 폴더 생성
    os.makedirs("./data", exist_ok=True)
    # 파일 경로 지정
    file_path = 'data/적금 리스트.txt'
    # 파일 경로에 불러온 데이터를 JSON 형태로 저장
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(json.dumps(data, ensure_ascii=False, indent=2))
    print(f"API에서 호출한 데이터가 '{file_path}'에 저장되었습니다.")

# JSON 데이터 로드 및 파싱
with open('./data/적금 리스트.txt', 'r', encoding='utf-8') as file:
    json_data = json.load(file)

class Document:
    def __init__(self, content):
        self.page_content = content  # 'text'를 'page_content'로 변경
        self.metadata = {}  # 기본적으로 빈 메타데이터 설정

# JSON 데이터를 문서로 변환
documents = [Document(content=f"{item['kor_co_nm']} {item['fin_prdt_nm']} {item['mtrt_int']} {item['spcl_cnd']} {item['join_member']} {item['etc_note']}") for item in json_data]

# 텍스트 분할기 초기화
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
split_data = text_splitter.split_documents(documents)

# Chroma 클라이언트 생성 및 사용
client = chromadb.PersistentClient()
answers = client.create_collection(name="answers")

# 사전 훈련된 모델 로드
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

# 임베딩 처리 및 저장
ids = []
metadatas = []
embeddings = []

for index, doc in enumerate(split_data):
    query = doc.page_content  # 'page_content' 속성 사용
    metadata = doc.metadata
    embedding = model.encode(query, normalize_embeddings=True)
    
    ids.append(str(index))
    metadatas.append(metadata)
    embeddings.append(embedding)

# 임베딩을 청크 단위로 추가
chunk_size = 100
total_chunks = len(embeddings) // chunk_size + 1
embeddings = [e.tolist() for e in embeddings]

for chunk_idx in range(total_chunks):
    start_idx = chunk_idx * chunk_size
    end_idx = (chunk_idx + 1) * chunk_size
    
    chunk_embeddings = embeddings[start_idx:end_idx]
    chunk_ids = ids[start_idx:end_idx]
    chunk_metadatas = metadatas[start_idx:end_idx]
    
    answers.add(embeddings=chunk_embeddings, ids=chunk_ids, metadatas=chunk_metadatas)
