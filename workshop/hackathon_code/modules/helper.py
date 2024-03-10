from functools import partial

from langchain.chains import LLMChain
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.chains.combine_documents import collapse_docs, split_list_of_docs
from langchain.schema.prompt_template import format_document
# from workshop.hackathon_code import prompts

# from prompts import QA_CHAIN_PROMPT, MAP_PROMPT, REDUCE_PROMPT


document_prompt = PromptTemplate.from_template("{page_content}")
partial_format_document = partial(format_document, prompt=document_prompt)


def format_docs(docs):
    return "\n\n".join(partial_format_document(doc) for doc in docs)


def map_reduce(doc, map_prompt, reduce_prompt) -> str:
    """
    We use the map-reduce pattern to answer the question from huge documents.
    """
    map_prompt = PromptTemplate.from_template(map_prompt)
    reduce_prompt = PromptTemplate.from_template(reduce_prompt)

    llm = llm=Bedrock(
        credentials_profile_name="default",
        # sets the profile name to use for AWS credentials (if not the default)
        region_name="us-east-1",  # sets the region name (if not the default)
        # model_id="ai21.j2-ultra-v1",  # set the foundation model
        model_id="anthropic.claude-v2:1"
        # model_id = GPT4_DEFAULT
    )
    # verbose=True

    # The chain we'll apply to each individual document.
    # Returns a summary of the document.

    map_chain = (
            {"context": partial_format_document}
            | map_prompt
            | llm
    )

    # A wrapper chain to keep the original Document metadata
    map_as_doc_chain = (
            RunnableParallel({"doc": RunnablePassthrough(), "content": map_chain})
            | (lambda x: Document(page_content=x["content"], metadata=x["doc"].metadata))
    ).with_config(run_name="Summarize (return doc)")
    #
    # collapse_chain = (
    #         {"context": doc}
    #         | reduce_prompt
    #         | general_llm
    # )

    # def get_num_tokens(docs):
    #     return general_llm.get_num_tokens(format_docs(docs))

    # The chain we'll repeatedly apply to collapse subsets of the documents
    # into a consolidate document until the total token size of our
    # documents is below some max size.

    # def collapse(
    #         docs,
    #         config,
    #         token_max=12000):
    #     collapse_ct = 1
    #     while get_num_tokens(docs) > token_max:
    #         config["run_name"] = f"Collapse {collapse_ct}"
    #         invoke = partial(collapse_chain.invoke, config=config)
    #         split_docs = split_list_of_docs(docs, get_num_tokens, token_max)
    #         docs = [collapse_docs(_docs, invoke) for _docs in split_docs]
    #         collapse_ct += 1
    #     return docs

    # The chain we'll use to combine our individual document summaries
    # (or summaries over subset of documents if we had to collapse the map results)
    # into a final summary.

    reduce_chain = (
            {"context": format_docs}
            | reduce_prompt
            | llm
    ).with_config(run_name="Reduce")

    # map_reduce = (
    #         map_as_doc_chain.map()
    #         | reduce_chain
    # ).with_config(run_name="Map reduce"
    #               )
    # The final full chain
    map_reduce = (map_as_doc_chain.map() | reduce_chain).with_config(
        run_name="Map reduce"
    )

    return map_reduce.invoke(doc, config={"max_concurrency": 30})