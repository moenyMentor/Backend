from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import WatsonxLLM
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from chatbot_finance_guide.config import PROJECT_ID, WML_CREDENTIALS

def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS",
    )

def load_llm_model():
    parameters = {
        GenParams.DECODING_METHOD: DecodingMethods.GREEDY.value,
        GenParams.MIN_NEW_TOKENS: 1,
        GenParams.MAX_NEW_TOKENS: 3000,
        GenParams.STOP_SEQUENCES: ["<|endoftext|>"]
    }
    
    model_id = 'mistralai/mistral-large'
    return WatsonxLLM(
        model_id=model_id,
        url=WML_CREDENTIALS.get("url"),
        apikey=WML_CREDENTIALS.get("apikey"),
        project_id=PROJECT_ID,
        params=parameters
    )