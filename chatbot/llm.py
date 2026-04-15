from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv
from langchain_upstage import UpstageEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

load_dotenv()

@st.cache_resource
def get_database():
    upstage_embedding = UpstageEmbeddings(model="solar-embedding-1-large")
    database = PineconeVectorStore(
        embedding=upstage_embedding,
        index_name="tax-index"
    )
    return database

database = get_database()

def get_llm(model="claude-haiku-4-5"):
    llm = ChatAnthropic(model = model)
    return llm


def get_retrieved_docs(user_message):
    retrieved_docs = database.similarity_search(user_message, k=3) # query에 적합하게 임베딩된 결과
    return retrieved_docs

def get_new_question(user_message):
    
    llm = get_llm()

    dictionary = ["사람을 나타내는 표현 -> 거주자"]
    dictionary_prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        사전: {dictionary}

        질문: {{question}}
        """
    )

    newQuery_chain = dictionary_prompt | llm | StrOutputParser()
    new_question = newQuery_chain.invoke({"question": user_message})
    return new_question

def get_qa_prompt(retrieved_docs, new_question):
    prompt = f"""[Identity]
    - 당신은 최고의 한국 소득세 전문가 입니다.
    - [Context]를 참고해서 사용자의 질문에 답변해주세요
    - 결과만 출력해주세요

    [Context]
    {retrieved_docs}

    Question: {new_question}
    """

def  get_ai_message(user_message):
    retrieved_docs = get_retrieved_docs(user_message) # query에 적합하게 임베딩된 결과
    new_question = get_new_question(user_message)
    qa_prompt = get_qa_prompt(retrieved_docs, new_question)
    ai_message = llm.invoke(prompt)
    return ai_message.content
