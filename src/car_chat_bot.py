import sys
import json
from src.utils.chatbot_utils import question_answering_prompt, parse_retriever_input, extraction, data_loader, vector_retrieval, write_user_data, clear_json_file
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document
import os
import getpass
from langchain_openai import ChatOpenAI
import re
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Dict
from langchain_core.runnables import RunnablePassthrough
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
import openai
import gradio as gr
import dotenv
dotenv.load_dotenv()
sys.path.insert(0, './src')


def car_chatbot():

    markdown_path = "Car-models-overview.md"
    llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.2)

    car_documents = data_loader(markdown_path)

    retriever = vector_retrieval(car_documents)

    document_chain = create_stuff_documents_chain(
        llm, question_answering_prompt)

    retrieval_chain = RunnablePassthrough.assign(
        context=parse_retriever_input | retriever,
    ).assign(
        answer=document_chain,
    )

    def predict(message, history):
        if not history:
            return "Hi! Please Enter your car Preferences?"
        history_langchain_format = []
        for human, ai in history:
            history_langchain_format.append(HumanMessage(content=human))
            history_langchain_format.append(AIMessage(content=ai))
        history_langchain_format.append(HumanMessage(content=message))
        history_langchain_format.append(AIMessage(content=ai))
        while True:
            response = retrieval_chain.invoke(
                {
                    "messages": history_langchain_format,
                }
            )

            history_langchain_format.append(
                AIMessage(content=response['answer']))
            if "team" in response['answer']:
                message_data = response['messages']
                user_data = extraction(message_data)
                write_user_data(user_data)
            return response['answer']

    gr.ChatInterface(predict).launch(share=True, debug=True)
