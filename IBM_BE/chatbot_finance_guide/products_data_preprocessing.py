import os
from dotenv import load_dotenv
import requests
from collections import OrderedDict
import json
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm
import pandas as pd


# .env 파일에서 API 키와 URL 가져오기
load_dotenv()
api_key = os.getenv("FINANCE_API_KEY")
api_url = os.getenv("API_URL")

data_str = ''

def fetch_data(products_name,page_no):
    # 요청 헤더 설정
    api_url=f"http://finlife.fss.or.kr/finlifeapi/{products_name}ProductsSearch.json?auth={api_key}&topFinGrpNo=020000&pageNo={page_no}"
    print(api_url)
    try:
        # API 요청 보내기
        response = requests.get(api_url)
        response.raise_for_status() 

        if response.status_code == 200:
            return response.json()
        else:
            print(f"Request failed with status code {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

def process_data(data, keys_to_extract):
    results = []
    if data and 'result' in data and 'baseList' in data['result']:
        for item in data['result']['baseList']:
            result = []
            for key in keys_to_extract:
                value = item.get(key, '')
                value = value.replace('\n', ' ') if isinstance(value, str) else str(value)
                result.append(f"{key}: {value}")
            results.append(' | '.join(result))
    return results

def main(api_key,products_name):
    keys_to_extract = ["kor_co_nm", "fin_prdt_nm", "join_way", "mtrt_int", "spcl_cnd", "join_deny", "join_member", "max_limit"]
    all_results = OrderedDict()
    page_no = 1
    max_page_no = 1
    products_name = products_name

    while page_no <= max_page_no:
        data = fetch_data(products_name, page_no)
        if not data:
            break

        max_page_no = data['result'].get('max_page_no', max_page_no)
        results = process_data(data, keys_to_extract)

        for result in results:
            all_results[result] = None  # Using OrderedDict to maintain order and remove duplicates

        if not results:
            print(f"No new data on page {page_no}. Stopping.")
            break

        page_no += 1

    DATA_PATH = ',/chatbot_finance_guide'
    os.makedirs(DATA_PATH, exist_ok=True)
    file_path = f'{DATA_PATH}/{products_name}_list.txt'
    with open(file_path, 'w', encoding='utf-8') as file:
        for line in all_results.keys():
            file.write(line + '\n')

    print(f"Total unique entries saved: {len(all_results)}")
    

if __name__ == "__main__":
    main(api_key,'saving')
    main(api_key,'deposit')
