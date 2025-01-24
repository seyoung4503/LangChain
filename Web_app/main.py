import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import load_prompt
from dotenv import load_dotenv
from langchain_teddynote import logging
import glob

# API KEY 정보로드
load_dotenv()
logging.langsmith("CH01-Basic")

st.title("LangGPT🐱‍👤✨")

# 처음 한번만 실행
if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성
    st.session_state["messages"] = []

with st.sidebar:
    clear_btn = st.button("대화 초기화")

    prompt_files = glob.glob("prompts/*.yaml")

    selected_prompt = st.selectbox("프롬프트 선택", prompt_files, index=0)
    task_input = st.text_input("TASK 입력", "")


# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메시지 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 체인 생성
def create_chain(prompt_filepath, task=""):
    # prompt | llm | output_parser

    # prompt 적용
    prompt = load_prompt(prompt_filepath, encoding="utf-8")

    if task:
        prompt = prompt.partial(task=task)

    print(prompt)

    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser
    return chain


if clear_btn:
    st.session_state["messages"] = []


print_messages()
# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

if user_input:
    # 웹에 출력
    st.chat_message("user").write(user_input)
    # 체인 생성
    chain = create_chain(selected_prompt, task=task_input)

    response = chain.stream({"question": user_input})
    with st.chat_message("assistant"):
        # 빈 컨테이너를 만들어서, 여기에 토큰을 스트리밍 출력
        container = st.empty()

        ai_answer = ""
        for token in response:
            ai_answer += token
            container.markdown(ai_answer)

    # 대화기록을 저장
    add_message("user", user_input)
    add_message("assistant", ai_answer)
