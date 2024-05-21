# Car Search Chat Bot

This repository contains the code and documentation for a sophisticated car search chatbot with advanced functionalities, leveraging vector databases and Large Language Models (LLMs). Below, you'll find a detailed overview of the project's functionalities and how to get started.

## Features

1. **Vector Creation for Car Documents**:
   - Create vectors for car documents to enable efficient and accurate document retrieval.

2. **Index Creation in ChromaDB**:
   - Organize and manage vectors by creating an index in ChromaDB and upserting (updating and inserting) these vectors locally.

3. **LLM Integration**:
   - Integrate a Large Language Model (LLM) to enhance the chatbot's ability to understand and generate human-like responses.

4. **Document Retriever Creation**:
   - Set up a retriever to fetch vectors from ChromaDB, ensuring relevant information is readily accessible.

5. **Integration with LangChain**:
   - Use the `create_stuff_documents_chain` function from LangChain to integrate our prompt and LLM, forming a cohesive unit.

6. **Retrieval Chain Construction**:
   - Create a retrieval chain to facilitate seamless communication between the LLM, document retriever, and the prompt.

7. **Conversation History Management**:
   - Use LangChain's `history_langchain_format` to store conversation history, allowing the chatbot to perform actions or generate messages based on previous human responses.

8. **Secondary LLM for User Information Extraction**:
   - After completing a conversation, pass all messages to a second LLM to extract the user's name and email, validate it, and provide the output in the desired format.

9. **Local Storage of Information**:
   - Store the extracted and validated user information in a local directory.

## Getting Started

## Build Image
- docker build --build-arg OPENAI_API_KEY="api-key" -t chatbot .
## Run Docker image
- docker run chatbot
### Prerequisites

- Python 3.10
- Required Python libraries (detailed in `requirements.txt`)
