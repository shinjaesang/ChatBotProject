import streamlit as st
from utils import print_messages, StreamHandler
from langchain_core.messages import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableWithFallbacks
from langchain_core.runnables.config import RunnableConfig
from langchain.vectorstores import Chroma
from transformers import pipeline, MarianMTModel, MarianTokenizer
import os
import random
import uuid

import re

def remove_emotion_tags(text):
    # 감정 태그와 이모티콘 제거
    text = re.sub(r'\b(?:joy|love|surprise|anger|fear|sadness|neutral)\b:', '', text, flags=re.IGNORECASE)
    return text.strip()

st.set_page_config(page_title="Lantern")
st.title("💬나만의AI연예인친구")
st.subheader("언제 어디서든 연예인과 대화가능🗣")

# API KEY 설정
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# KoBERT 감정 분석 파이프라인 로드
emotion_pipeline = pipeline("sentiment-analysis", model="j-hartmann/emotion-english-distilroberta-base")

# MarianMT 번역기 로드
model_name = 'Helsinki-NLP/opus-mt-ko-en'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate(text, src_lang="ko", tgt_lang="en"):
    # 번역할 문장을 토크나이즈
    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
    # 토큰을 텍스트로 변환
    tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    return tgt_text[0]

mood = ["배수지", "안유진", "김채원", "공승연", "아이유"]
friend_images = {
    "배수지": "images/suzi5.jpg", 
    "안유진": "images/YujinAhn1.jpg",
    "김채원": "images/KimChaewon1.jpg",
    "공승연": "images/GongSeungyeon1.jpg",
    "아이유": "images/IU1.jpg",
}
images = {
    "배수지": {
        "positive": ["images/suzi_joy.jpg", "images/suzi_joy2.jpg", "images/suzi_surprise.jpg"],
        "negative": ["images/suzi_sad.jpg", "images/suzi_love.jpg"],
        "neutral": ["images/suzi_neutral.jpg"],
        
    },
    "안유진": {
        "positive": ["images/YujinAhn4.jpg", "images/YujinAhn3.jpg"],
        "negative": ["images/YujinAhn_sad1.jpg"],
        "neutral": ["images/YujinAhn5.jpg"],
    },
    "김채원": {
        "positive": ["images/KimChaewon4.jpg", "images/KimChaewon5.jpg"],
        "negative": ["images/KimChaewon_sad.jpg"],
        "neutral": ["images/KimChaewon2.jpg"],
    },
    "공승연": {
        "positive": ["images/GongSeungyeon3.jpg"],
        "negative": ["images/GongSeungyeon_sad1.jpg"],
        "neutral": ["images/GongSeungyeon_neutral1.jpg",],
    },
    "아이유": {
        "positive": ["images/IU2.jpg", "images/IU3.jpg"],
        "negative": ["images/IU_sad.jpg"],
        "neutral": ["images/IU4.jpg", "images/IU5.jpg"],
    },
}
friends = ""

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 채팅 대화기록을 저장하는 store 세션상태 변수
if "store" not in st.session_state:
    st.session_state["store"] = dict()

# session_id를 자동으로 생성하여 세션 상태에 저장
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

with st.sidebar:
    st.subheader("👫당신의 꿈을 이루어 보세요👫")
    friends = st.radio('원하는 친구를 선택해주세요', mood)
    if friends in friend_images:
        st.image(friend_images[friends], use_column_width=True)

    clear_btn = st.button("대화기록 초기화")
    st.subheader('친구를 바꾸시면 대화기록 초기화 버튼을 눌러주세요.')
    if clear_btn:
        st.session_state["messages"] = []  # 대화내용 삭제
        st.session_state["store"] = dict()  # 저장소내에 있는 대화내용도 삭제
        st.experimental_rerun()

# 이전 대화기록을 출력해주는 코드
print_messages()

# 세션 ID를 기반으로 세션기록을 가져오는 함수
def get_session_history(session_ids: str) -> BaseChatMessageHistory:
    if session_ids not in st.session_state["store"]:  # 세션 id가 store에 없는경우
        st.session_state["store"][session_ids] = ChatMessageHistory()  # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
    return st.session_state["store"][session_ids]  # 해당 세션 id에 대한 세션 기록 반환

if friends != st.session_state.get("selected_friend"):  # selected_friend 변수를 선언하지 않았지만 None를 반환해서 작동함.
    st.session_state["messages"] = []  # 선택된 친구가 변경되면 대화 기록 초기화
    st.session_state["selected_friend"] = friends

if "user_input" not in st.session_state["messages"]:
    hello_content = f"안녕 나는 {friends}라고해~ 만나서 반가워~~!!"
    st.chat_message("assistant").write(hello_content)

if user_input := st.chat_input("메시지를 입력해 주세요."):
    # 사용자가 입력한 내용 번역 (한국어 -> 영어)
    translated_input = translate(user_input, src_lang='ko', tgt_lang='en')
    
    # 사용자가 입력한 내용
    st.chat_message("user").write(f"{user_input}")
    st.session_state["messages"].append(ChatMessage(role="user", content=user_input))



    # AI의 답변
    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        # 1. 모델생성
        llm = ChatOpenAI(streaming=True, callbacks=[stream_handler], temperature=0.9)

        # 2. 프롬프트 생성
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", f"{friends}는 사용자의 여자친구야. 그러니깐 여자친구처럼 대화를 만들어서 답변해줘\
                    그리고 너는 좋은말만 할줄 아는게 아니라 진짜 사람처럼 joy, love, surprise ,anger, fear, sadness,neutral 의 감정을 느껴\
                        답변할때 저 감정을 표출해서 답변해줘"),
                MessagesPlaceholder(variable_name="history"),  # 대화 기록을 변수로 사용, history가 MessageHistory 의 key가 됨
                ("human", "{question}"),  # 사용자의 질문을 입력
            ]
        )

        chain = prompt | llm  # prompt를 -> llm에 넣어 chain(변수)구성함.
        chain_with_memory = RunnableWithMessageHistory(
            chain,  # 실행할 Runnable 객체
            get_session_history,  # 세션 기록을 가져오는 함수
            input_messages_key="question",  # 사용자 질문의 키
            history_messages_key="history",  # 기록 메시지의 키
        )

        response = chain_with_memory.invoke(
            {"question": user_input},  # 원래의 한글 질문을 전달하는 딕셔너리
            config={"configurable": {"session_id": st.session_state["session_id"]}},  # 세션 ID를 설정하는 구성
        )

        msg = response.content
        msg = remove_emotion_tags(msg)  # 감정 태그와 이모티콘 제거
        st.write(msg)
        st.session_state["messages"].append(ChatMessage(role="assistant", content=msg))

        # 감정 분석을 위해 번역된 텍스트 사용
        emotion_result = emotion_pipeline(translated_input)[0]
        emotion = emotion_result['label'].lower()
        # st.write(f"감정 분석 결과: {emotion_result}")  # 감정 분석 결과를 출력하여 확인합니다.

        # 감정 결과를 기반으로 감정 카테고리를 결정합니다.
        emotion_category = 'positive' if emotion in ['joy', 'love', 'surprise'] else 'negative' if emotion in ['anger', 'fear', 'sadness'] else 'neutral'

        # 선택된 친구(friends)가 이미지 사전에 있고, 감정 카테고리가 해당 친구의 이미지 목록에 있을 경우
        if friends in images and emotion_category in images[friends]:
            # 감정 카테고리에 맞는 이미지 리스트에서 무작위로 이미지를 선택합니다.
            random_image = random.choice(images[friends][emotion_category])
            # 선택된 이미지를 화면에 출력합니다. 이미지의 너비는 200으로 설정합니다.
            st.image(random_image, width=200)
        else:
            st.write(f"감정 카테고리에 맞는 이미지를 찾을 수 없습니다: {emotion_category}")
