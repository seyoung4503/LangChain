import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import load_prompt
from dotenv import load_dotenv
from langchain_teddynote import logging
import glob
import os


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# API KEY 정보로드
load_dotenv()
logging.langsmith("CH01-Basic")

# 캐시 디렉토리 설정 (.폴더는 숨김 폴더)
if not os.path.exists(".cache"):
    os.mkdir(".cache")

# 파일 업로드 전용 폴더
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")


st.title("PDF 기반 QA😎")

# 처음 한번만 실행
if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성
    st.session_state["messages"] = []

if "chain" not in st.session_state:
    st.session_state["chain"] = None

with st.sidebar:
    clear_btn = st.button("대화 초기화")

    # 파일 업로드
    uploaded_file = st.file_uploader("Choose a file", type=["pdf"])

    selected_prompt = "prompts/pdf-rag.yaml"


# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메시지 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 파일을 캐시 저장(시간이 오래 걸리는 작업을 처리할 예정)
@st.cache_resource(
    show_spinner="업로드한 파일을 처리 중입니다..."
)  # decorator -> 파일을 캐싱(캐싱한 값을 사용)
def embed_file(file):
    # 업로드 파일을 캐시 디렉토리에 저장
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    # 단계 1: 문서 로드(Load Documents)
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()

    # 단계 2: 문서 분할(Split Documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_documents = text_splitter.split_documents(docs)

    # 단계 3: 임베딩(Embedding) 생성
    embeddings = OpenAIEmbeddings()

    # 단계 4: DB 생성(Create DB) 및 저장
    # 벡터스토어를 생성합니다.
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    # 단계 5: 검색기(Retriever) 생성
    # 문서에 포함되어 있는 정보를 검색하고 생성합니다.
    retriever = vectorstore.as_retriever()

    return retriever


# 체인 생성
def create_chain(retriever):
    # prompt | llm | output_parser

    # prompt 적용
    # prompt = load_prompt(prompt_filepath, encoding="utf-8")

    # 단계 6: 프롬프트 생성(Create Prompt)
    # 프롬프트를 생성합니다.
    prompt = PromptTemplate.from_template(
        """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Answer in Korean.

    #Context: 
    {context}

    #Question:
    {question}

    #Answer:"""
    )

    # 단계 7: 언어모델(LLM) 생성
    # 모델(LLM) 을 생성합니다.
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    # 단계 8: 체인(Chain) 생성
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# 파일 업로드 되었을 때
if uploaded_file:
    # 파일 업로드 후 retriever 생성 (작업생성 오래걸림)
    retriever = embed_file(uploaded_file)
    chain = create_chain(retriever)
    st.session_state["chain"] = chain


if clear_btn:
    st.session_state["messages"] = []


print_messages()
# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

# 경고 메시지를 띄우기 위한 빈 영역
warning_message = st.empty()

if user_input:

    # 체인 생성
    chain = st.session_state["chain"]

    if chain is not None:

        # 웹에 출력
        st.chat_message("user").write(user_input)

        response = chain.stream(user_input)
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

    else:
        # 파일을 업로드 하라는 경고 메시지 출력
        warning_message.error("파일을 업로드 해주세요")
