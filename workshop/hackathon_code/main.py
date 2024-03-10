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
from langchain.chains.llm import LLMChain
from langchain.llms.bedrock import Bedrock

from langchain.document_loaders import CSVLoader

from modules.prompts import QA_CHAIN_PROMPT, CONDENSE_QUESTION_PROMPT, \
    q_generator_parser, INITIAL_CHAIN_PROMPT, FILTER_GEN_PROMPT

# import langsmith to debug easier

_ = load_dotenv("credentials.env")

try:  # for local debug
    os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
    os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT")
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

except:
    pass

memory = []

#
# def create_pages_from_documents(documents, n):
#     """
#     Groups every 'n' documents into a 'page_content' key of a new dictionary.
#
#     Parameters:
#     - documents: List of 'Document' objects or similar structures.
#     - n: Number of documents to group into each 'page_content'.
#
#     Returns:
#     - List[Dict]: A list where each element is a dictionary with a 'page_content' key.
#     """
#     pages = []  # Resulting list of page dictionaries
#     current_page_content = []  # Temporary storage for the current page's content
#
#     for document in documents:
#         current_page_content.append(document)  # Add the document to the current page
#         if len(current_page_content) == n:  # If the page is full
#             pages.append({'page_content': current_page_content})  # Add the page to the list
#             current_page_content = []  # Start a new page
#
#     # Add the last page if it has less than 'n' documents but is not empty
#     if current_page_content:
#         pages.append({'page_content': current_page_content})
#
#     return pages


general_llm = LLMChain(
    llm=Bedrock(
        credentials_profile_name="default",
        # sets the profile name to use for AWS credentials (if not the default)
        region_name="us-east-1",  # sets the region name (if not the default)
        # model_id="ai21.j2-ultra-v1",  # set the foundation model
        model_id="anthropic.claude-v2:1",
        model_kwargs={"temperature": 0}
        # model_id = GPT4_DEFAULT
    ),
    prompt=QA_CHAIN_PROMPT,
    # verbose=True
)

standalone_question_gen = LLMChain(
    llm=Bedrock(
        credentials_profile_name="default",
        # sets the profile name to use for AWS credentials (if not the default)
        region_name="us-east-1",  # sets the region name (if not the default)
        # model_id="ai21.j2-ultra-v1",  # set the foundation model
        model_id="anthropic.claude-v2:1",
        model_kwargs={"temperature": 0}
        # model_id = GPT4_DEFAULT
    ),
    prompt=CONDENSE_QUESTION_PROMPT,
    output_parser=q_generator_parser
)

initial_chain = LLMChain(
    llm=Bedrock(
        credentials_profile_name="default",
        # sets the profile name to use for AWS credentials (if not the default)
        region_name="us-east-1",  # sets the region name (if not the default)
        # model_id="ai21.j2-ultra-v1",  # set the foundation model
        model_id="anthropic.claude-v2:1",
        model_kwargs={"temperature": 0}
        # model_id = GPT4_DEFAULT
    ),
    prompt=INITIAL_CHAIN_PROMPT
)

filter_gen_chain = LLMChain(
    llm=Bedrock(
        credentials_profile_name="default",
        # sets the profile name to use for AWS credentials (if not the default)
        region_name="us-east-1",  # sets the region name (if not the default)
        # model_id="ai21.j2-ultra-v1",  # set the foundation model
        model_id="anthropic.claude-v2:1",
        model_kwargs={"temperature": 0}
        # model_id = GPT4_DEFAULT
    ),
    prompt=FILTER_GEN_PROMPT
)

# input_text = "Which models have the ability of steam cooking?"

# standalone_question = standalone_question_gen.run(chat_history=chat_history,
#                                                   question=input_text)["standalone_question"]

# response = get_rag_chat_response(input_text=input_text,
#                                  memory=get_memory(),
#                                  index=get_index(),
#                                  prompt=QA_CHAIN_PROMPT)
doc_path = "docs/dataset_words.csv"

loader = CSVLoader(file_path=doc_path)

data = loader.load()
# print(data[:5])

# documents = data  # Your list of 'Document' objects as loaded or defined
# n = 15  # Example: Group every 5 documents
#
# pages = create_pages_from_documents(documents, n)

# context = pd.read_csv(doc_path)
# response = general_llm.run(query = input_text,
#                    context = data[:15]
# )

# I want 'data' to be a list of dict variables

# response = map_reduce(pages[3].get("page_content"),
#                       f"{map_prompt}{input_text}#####",
#                       f"{reduce_prompt}{input_text}#####")
#
# print(response)

def give_it_to_me_baby(prompt, memory):

    # Generate the standalone question
    query = standalone_question_gen.run(chat_history=memory,
                                        question=prompt)["standalone_question"]

    columns = initial_chain.run(query=query)

    filter = filter_gen_chain.run(query=query,
                                  columns=columns)

    result = filter
    return result

print(give_it_to_me_baby("Hey! I want an oven to replace my steam cooker in my bakery. I need it to be cheaper "
                         "than 3000 euros and compact. Actually I want it to have the ability of cleaning itself", []))