import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID", None)
WML_CREDENTIALS = {
    "apikey": os.getenv("API_KEY", None),
    "url": 'https://us-south.ml.cloud.ibm.com'
}