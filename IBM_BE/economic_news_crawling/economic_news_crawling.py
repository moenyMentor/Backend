from bs4 import BeautifulSoup
import requests

def news_crawling():
    # 네이버 경제 뉴스 url
    url = "https://news.naver.com/section/101"
    
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to retrieve news page. Status code: {response.status_code}")
    
    # HTML 파싱
    soup = BeautifulSoup(response.text, "html.parser")
    
    # 뉴스 헤드라인 데이터 추출 (상위 3개 뉴스만)
    titles = [title.get_text() for title in soup.select(".sa_text_strong")[:3]]
    contents = [content.get_text() for content in soup.select(".sa_text_lede")[:3]]
    urls = [link.get("href") for link in soup.select(".sa_text > a")[:3]]

    # 결과를 딕셔너리 형태로 반환
    return {
        "title": titles,
        "content": contents,
        "url": urls
    }

'''사용 예시
from economic_news_crawling import news_crawling

# 뉴스 데이터를 가져오기
news_data = news_crawling()

# 뉴스 제목, 내용, URL 리스트에 접근
news_title = news_data["title"]
news_content = news_data["content"]
news_url = news_data["url"]

# 첫 번째 뉴스 제목, 내용, URL 출력
print(f"Title: {news_title[0]}")
print(f"Content: {news_content[0]}")
print(f"URL: {news_url[0]}")

'''