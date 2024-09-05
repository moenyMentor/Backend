from fastapi import FastAPI
from pydantic import BaseModel
from chatbot_economic_terms.rag_chain import create_rag_chain
from chatbot_finance_guide.RAG_Chain import create_rec_rag
from economic_news_crawling.economic_news_crawling import news_crawling

import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()



#CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React 앱 주소
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#메시지를 받을 데이터 모델 정의
class Message(BaseModel):
    message:str


#경제공부 챗봇
@app.post("/api/messages")
async def get_message(msg:Message):
    print(f"Received message:{msg.message}")
    user_input=msg.message
    response=create_rag_chain(user_input)
    print("status:success")
    print({response})
    return{"response":response}



#경제상품 추천 챗봇
@app.post("/api/recommend")
async def get_recommend(msg:Message):
    print(f"Received message:{msg.message}")
    user_input=msg.message
    response=create_rec_rag(user_input)
    print("status:success")
    print({response})
    return{"response":response}



#크롤링한 뉴스 가져오기
@app.get("/api/news")
async def get_news():
    news_data=news_crawling()
    return news_data
   

# class QueryRequest(BaseModel):
#     query: str

# @app.post("/studychat")
# async def study_response(request:QueryRequest):
#     user_input=request.query
#     response=create_rag_chain(user_input)
#     return{"response":response}

# if __name__=="__main__":
#     import uvicorn
#     uvicorn.run(app,host="0.0.0.0",port=8001)

