import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


# API KEY 정보로드
load_dotenv()

st.title("LangGPT")

# 처음 한번만 실행
if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성
    st.session_state["messages"] = []


# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)
        # st.write(f"{chat_message.role}: {chat_message.content}")


# 새로운 메시지 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


print_messages()
# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

if user_input:
    # 웹에 출력
    st.chat_message("user").write(user_input)
    st.chat_message("assistant").write(user_input)

    # 대화기록을 저장
    add_message("user", user_input)
    add_message("assistant", user_input)
