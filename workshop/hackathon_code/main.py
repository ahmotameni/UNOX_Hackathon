# Hello!
# I'm gonna build a chatbot with memory using RAG, for customers to ask their questions about the products and services of a company.

# Step 1: Integrate the data
# - We have to build vector embeddings for the data

# Step 2: Get the query and find the related document

# Step 3: Answer the question according to the document

# Step 4: If the answer is not in the document, ask the user to ask to the customer service

## Step 1

import os

from dotenv import load_dotenv
from langchain.memory import ConversationBufferWindowMemory
from langchain.llms.bedrock import Bedrock
from langchain.chains import ConversationalRetrievalChain

from langchain.embeddings import BedrockEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain.document_loaders import CSVLoader

from prompts import QA_CHAIN_PROMPT

# import langsmith to debug easier
import langchain
_ = load_dotenv("credentials.env")

try:  # for local debug
    os.environ["LANGCHAIN_TRACING_V2"] =os.getenv("LANGCHAIN_TRACING_V2")
    os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT")
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
except:
    pass



def get_llm(prompt):
    model_kwargs = {  # AI21
        "maxTokens": 8000,
        "temperature": 0,
        "topP": 0.3,
        "stopSequences": ["Human:"],
        "countPenalty": {"scale": 0},
        "presencePenalty": {"scale": 0},
        "frequencyPenalty": {"scale": 0},
        "prompt": prompt
    }

    llm = Bedrock(
        credentials_profile_name="default",
        # sets the profile name to use for AWS credentials (if not the default)
        region_name="us-east-1",  # sets the region name (if not the default)
        model_id="ai21.j2-ultra-v1",  # set the foundation model
        model_kwargs=model_kwargs)  # configure the properties for Claude

    return llm


def get_index():  # creates and returns an in-memory vector store to be used in the application

    embeddings = BedrockEmbeddings(
        credentials_profile_name="default",
        # sets the profile name to use for AWS credentials (if not the default)
        region_name="us-east-1",  # sets the region name (if not the default)
    )  # create a Titan Embeddings client

    doc_path = "docs/fakedata.csv"  # assumes local PDF file with this name

    loader = CSVLoader(file_path=doc_path)

    text_splitter = RecursiveCharacterTextSplitter(  # create a text splitter
        separators=["\n\n", "\n", ".", " "],
        # split chunks at (1) paragraph, (2) line, (3) sentence, or (4) word, in that order
        chunk_size=1000,  # divide into 1000-character chunks using the separators above
        chunk_overlap=100  # number of characters that can overlap with previous chunk
    )

    index_creator = VectorstoreIndexCreator(  # create a vector store factory
        vectorstore_cls=FAISS,  # use an in-memory vector store for demo purposes
        embedding=embeddings,  # use Titan embeddings
        text_splitter=text_splitter,  # use the recursive text splitter
    )

    index_from_loader = index_creator.from_loaders([loader])  # create a vector store index from the loaded PDF

    return index_from_loader  # return the index to be cached by the client app


def get_memory():  # create memory for this chat session

    memory = ConversationBufferWindowMemory(memory_key="chat_history",
                                            return_messages=True)  # Maintains a history of previous messages

    return memory


def get_rag_chat_response(input_text, memory, index, prompt):  # chat client function

    llm = get_llm(prompt=prompt)

    conversation_with_retrieval = ConversationalRetrievalChain.from_llm(llm, index.vectorstore.as_retriever(),
                                                                        memory=memory)

    chat_response = conversation_with_retrieval(
        {"question": input_text})  # pass the user message and summary to the model

    return chat_response['answer']


input_text = "Which models are suitable for making ribs?"

response = get_rag_chat_response(input_text=input_text,
                                 memory=get_memory(),
                                 index=get_index(),
                                 prompt=QA_CHAIN_PROMPT)

print(response)