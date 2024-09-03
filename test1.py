# 데이터 로드와 텍스트 분할
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm

# 데이터 로드
loader = TextLoader("./data/적금 리스트.txt", encoding='utf-8')
api_data = loader.load()

# 텍스트 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
split_data = text_splitter.split_documents(api_data)

# 모델 로드
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

# 클라이언트와 컬렉션 생성
client = chromadb.PersistentClient()
answers = client.create_collection(name="answers")

# 데이터 삽입
ids = []
metadatas = []
embeddings = []

for idx, doc in enumerate(split_data):
    query = doc["text"]  # document에서 필요한 데이터를 추출
    metadata = {"query": query}
    
    embedding = model.encode(query, normalize_embeddings=True)
    
    ids.append(str(idx))
    metadatas.append(metadata)
    embeddings.append(embedding)
    
chunk_size = 100
total_chunks = len(embeddings) // chunk_size + 1
embeddings = [e.tolist() for e in tqdm(embeddings)]

for chunk_idx in tqdm(range(total_chunks)):
    start_idx = chunk_idx * chunk_size
    end_idx = (chunk_idx + 1) * chunk_size
    
    chunk_embeddings = embeddings[start_idx:end_idx]
    chunk_ids = ids[start_idx:end_idx]
    chunk_metadatas = metadatas[start_idx:end_idx]
    
    answers.add(embeddings=chunk_embeddings, ids=chunk_ids, metadatas=chunk_metadatas)
