# Hello!
# I'm gonna build a chatbot with memory using RAG, for customers to ask their questions about the products and services of a company.

# Step 1: Integrate the data
# - We have to build vector embeddings for the data

# Step 2: Get the query and find the related document

# Step 3: Answer the question according to the document

# Step 4: If the answer is not in the document, ask the user to ask to the customer service

## Step 1

import os

import pandas as pd
from dotenv import load_dotenv
from langchain.chains.llm import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.llms.bedrock import Bedrock
from langchain.chains import ConversationalRetrievalChain

from langchain.embeddings import BedrockEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain.document_loaders import CSVLoader

from prompts import QA_CHAIN_PROMPT, qa_template, MAP_PROMPT, REDUCE_PROMPT
from modules.helper import map_reduce

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


def create_pages_from_documents(documents, n):
    """
    Groups every 'n' documents into a 'page_content' key of a new dictionary.

    Parameters:
    - documents: List of 'Document' objects or similar structures.
    - n: Number of documents to group into each 'page_content'.

    Returns:
    - List[Dict]: A list where each element is a dictionary with a 'page_content' key.
    """
    pages = []  # Resulting list of page dictionaries
    current_page_content = []  # Temporary storage for the current page's content

    for document in documents:
        current_page_content.append(document)  # Add the document to the current page
        if len(current_page_content) == n:  # If the page is full
            pages.append({'page_content': current_page_content})  # Add the page to the list
            current_page_content = []  # Start a new page

    # Add the last page if it has less than 'n' documents but is not empty
    if current_page_content:
        pages.append({'page_content': current_page_content})

    return pages


llm = LLMChain(
    llm = Bedrock(
        credentials_profile_name="default",
        # sets the profile name to use for AWS credentials (if not the default)
        region_name="us-east-1",  # sets the region name (if not the default)
        # model_id="ai21.j2-ultra-v1",  # set the foundation model
        model_id = "anthropic.claude-v2:1"
        # model_id = GPT4_DEFAULT
        ),
    prompt = QA_CHAIN_PROMPT,
    verbose = True
)


input_text = "Which models weigh between 40 to 60 KG?"

# response = get_rag_chat_response(input_text=input_text,
#                                  memory=get_memory(),
#                                  index=get_index(),
#                                  prompt=QA_CHAIN_PROMPT)
doc_path = "docs/data.csv"

loader = CSVLoader(file_path=doc_path)

data = loader.load()
# print(data[:5])

documents = data  # Your list of 'Document' objects as loaded or defined
n = 5  # Example: Group every 5 documents

pages = create_pages_from_documents(documents, n)

# context = pd.read_csv(doc_path)
# response = llm.run(query = input_text,
#                    context = data[:15]
# )

# I want 'data' to be a list of dict variables

print(pages[0].get("page_content"))

response = map_reduce(pages[0].get("page_content"), MAP_PROMPT, REDUCE_PROMPT)

print(response)