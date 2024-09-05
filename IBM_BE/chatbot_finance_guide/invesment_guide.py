from chatbot_finance_guide.model import load_llm_model
from chatbot_finance_guide.sys_prompt import guide_prompt

def create_guide():

    #사용자 입력받을 변수
    age = int(input("나이를 입력해주세요:"))
    salary = input("연봉을 입력해주세요:")
    goal =  input("목표(기간 포함)을 입력해주세요:")
    seed_money =  input("현재 시드머니를 입력해주세요:")
    investment_exp = input("투자 경험을 입력해주세요:")
    investment_tendency = input("투자 유형을 입력해주세요:")
    available_amount = input("월별 가용 금액을 입력해주세요:")

    prompt = guide_prompt(age, salary, goal, seed_money, investment_exp, investment_tendency, available_amount)
    llm = load_llm_model()
    
    response = llm.generate(prompts=[prompt])

    return response.generations[0][0].text