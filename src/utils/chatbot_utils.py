import json
from openai import OpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
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
import openai


SYSTEM_TEMPLATE = """
You have to Engage in a conversation with the user, asking follow-up questions based on their previous responses.
-Your initial answer must be greeting someone and asking about car prefrences.
-As you are car Expert , you have to suggest two vehicles in answer on the bases of given context and given features of the cars available in the context only, not from your own knowledge.
-Propose 2 car models at a time, and specify the reason why they are the best option, based on the user's preferences provided in content.
-Reason should according to user preferences provided in content.
-After your answer from context throw a message "would you like to buy car from above options"
-You have to identify if user enter car specification then throw a message "would you like to buy any of above car".
-If user answer in no or any other text then again throw a message "Please Enter your car Preferences".
-If user answer in yes for buying a car then throw a message "Please enter the name of specific car".
After above message  throw a message "please enter your name".
After above message  throw a message "please enter your email".
After above message Last message will be "Our team will get back to you soon"

you should response in format as :
car name: car name
car model: car model
Reason : Reasoning
Answer the user's questions based on the below context.
If the context doesn't contain any relevant information to the question, don't make something up and just say "sorry I don't know it is out of my knowledge":

Note: The answer must be from context don't use your knowledge while answering.
<context>
{context}
</context>

"""

question_answering_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            SYSTEM_TEMPLATE,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


def parse_retriever_input(params: Dict):
    return params["messages"][-1].content


def extraction(list_of_messages):
    message_string = str(list_of_messages)
    all_messages = f"'{message_string}'"

    client = OpenAI()
    prompt = """ You are an extractor that will analyze the data given below context and extract name of the Human , his Email and name of the car that Human selected in the given data. 

 Follow instructions carefully:
  1. get desired data only from provided data
  2. Data have a chat of Human and AI , carefully get the desired response provided by Human. 
  3. validate provided Email if the format is correct then return else mention invalid 
  4. Output data in a structured format

  context: {all_messages}
"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": all_messages}
        ],
        stop=None,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


def data_loader(markdown_path):
    loader = UnstructuredMarkdownLoader(markdown_path)
    data = loader.load()
    page_content = data[0].page_content
    car_pattern = r'(\d+\.\s+\w+\s+\w+)(.*?)(?=\d+\.\s+\w+\s+\w+|\Z)'
    car_entries = re.findall(car_pattern, page_content, re.DOTALL)
    car_documents = []
    for car_entry in car_entries:
        car_document_content = car_entry[0] + car_entry[1]
        car_documents.append(Document(
            page_content=car_document_content, metadata={'source': markdown_path}))

    return car_documents


def vector_retrieval(car_documents):
    persist_directory = "./chroma_db"
    embedding_function = OpenAIEmbeddings()

    if os.path.exists(persist_directory):
        vectorstore = Chroma(persist_directory=persist_directory,
                             embedding_function=embedding_function)
    else:
        vectorstore = Chroma.from_documents(
            documents=car_documents, embedding=embedding_function, persist_directory=persist_directory)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    return retriever


def write_user_data(user_data):
    user_data = json.loads(user_data)
    with open('user_data/user_data.json', 'r') as file:
        data = json.load(file)
    data.append(user_data)
    with open('user_data/user_data.json', 'w') as f:
        json.dump(data, f)


def clear_json_file(filename='user_data/user_data.json'):
    with open(filename, 'w') as f:
        json.dump([], f)
