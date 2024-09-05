from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from chatbot_economic_terms.sys_prompt import prompt
from chatbot_economic_terms.database import load_vector_db
from chatbot_economic_terms.model import load_llm_model

db = load_vector_db()
retriever = db.as_retriever(search_kwargs={'k': 5})
prompt_content = prompt 
llm = load_llm_model()

def create_rag_chain(user_input):
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_content
        | llm
        | StrOutputParser()
    )

    response = rag_chain.invoke(user_input)
    return response
    # return rag_chain


# #### 테스트 ####
# chain = create_rag_chain()
print(create_rag_chain("금리가 무엇인가요?"))
# # print(retriever.get_relevant_documents("금리가 무엇인가요?"))

# db = load_vector_db()
# print(db.get())


