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
from langchain.llms.bedrock import Bedrock

from langchain.document_loaders import CSVLoader

from modules.prompts import QA_CHAIN_PROMPT, CONDENSE_QUESTION_PROMPT, \
    q_generator_parser, initial_prompt, filter_chain_prompt, CODE_GEN_PROMPT, filter_parser, answer_chain_prompt

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
    prompt=initial_prompt
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
    prompt=filter_chain_prompt,
    output_parser=filter_parser
)

answer_chain = LLMChain(
    llm=Bedrock(
        credentials_profile_name="default",
        # sets the profile name to use for AWS credentials (if not the default)
        region_name="us-east-1",  # sets the region name (if not the default)
        # model_id="ai21.j2-ultra-v1",  # set the foundation model
        model_id="anthropic.claude-v2:1",
        model_kwargs={"temperature": 0}
        # model_id = GPT4_DEFAULT
    ),
    prompt=answer_chain_prompt
)

code_gen_chain = LLMChain(
    llm=Bedrock(
        credentials_profile_name="default",
        # sets the profile name to use for AWS credentials (if not the default)
        region_name="us-east-1",  # sets the region name (if not the default)
        # model_id="ai21.j2-ultra-v1",  # set the foundation model
        model_id="anthropic.claude-v2:1",
        model_kwargs={"temperature": 0}
        # model_id = GPT4_DEFAULT
    ),
    prompt=CODE_GEN_PROMPT
)

doc_path = "docs/final_dataset.csv"

# loader = CSVLoader(file_path=doc_path)
#
# data = loader.load()

def filter_data_using_filter_chain(conditions, data_path):

    df = pd.read_csv(data_path)
    df = df.query(' & '.join(conditions))
    # print(df)
    return df

def give_it_to_me_baby(prompt, memory):

    # Generate the standalone question
    query = standalone_question_gen.run(chat_history=memory,
                                        question=prompt)["standalone_question"]

    columns = initial_chain.run(query=query)

    # columns = str(columns)
    #
    print(columns)

    filters = filter_gen_chain.run(columns=columns,
                                   query=query)

    print(filters)

    filtered_df = filter_data_using_filter_chain(filters['filters'], doc_path)
    # print(filtered_df.head(1))

    result = answer_chain.run(context=filtered_df.head(1))
    # result = code_gen_chain.run(columns=columns,
    #                             filters=filters)
    return result

print(give_it_to_me_baby("Hey! I want an oven to replace my oven in my bakery. I need it to be less "
                         "than 3000 euros.", []))