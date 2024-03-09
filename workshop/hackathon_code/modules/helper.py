from langchain.prompts import PromptTemplate
from prompts import QA_CHAIN_PROMPT

def map_reduce(splits, template, reduce_template, model) -> dict:
    """
    We use the map-reduce pattern to answer the question from huge documents.
    """
    map_prompt = PromptTemplate.from_template(template)
    reduce_prompt = PromptTemplate.from_template(reduce_template)

    llm = ChatOpenAI(temperature=0, model=model)

    # The chain we'll apply to each individual document.
    # Returns a summary of the document.

    map_chain = (
            {"context": partial_format_document}
            | map_prompt
            | llm
            | StrOutputParser()
    )

    # A wrapper chain to keep the original Document metadata
    map_as_doc_chain = (
            RunnableParallel({"doc": RunnablePassthrough(), "content": map_chain})
            | (lambda x: Document(page_content=x["content"], metadata=x["doc"].metadata))
    ).with_config(run_name="Summarize (return doc)")

    collapse_chain = (
            {"context": format_docs}
            | reduce_prompt
            | llm
            | StrOutputParser()
    )

    def get_num_tokens(docs):
        return llm.get_num_tokens(format_docs(docs))

    # The chain we'll repeatedly apply to collapse subsets of the documents
    # into a consolidate document until the total token size of our
    # documents is below some max size.

    def collapse(
            docs,
            config,
            token_max=12000):
        collapse_ct = 1
        while get_num_tokens(docs) > token_max:
            config["run_name"] = f"Collapse {collapse_ct}"
            invoke = partial(collapse_chain.invoke, config=config)
            split_docs = split_list_of_docs(docs, get_num_tokens, token_max)
            docs = [collapse_docs(_docs, invoke) for _docs in split_docs]
            collapse_ct += 1
        return docs

    # The chain we'll use to combine our individual document summaries
    # (or summaries over subset of documents if we had to collapse the map results)
    # into a final summary.

    reduce_chain = (
            {"context": format_docs}
            | reduce_prompt
            | llm
            | StrOutputParser()
    ).with_config(run_name="Reduce")

    # The final full chain
    map_reduce = (
            map_as_doc_chain.map()
            | collapse
            | reduce_chain
    ).with_config(run_name="Map reduce"
                  )

    toc_chain = (
            {"context": format_str}
            | toc_prompt
            | llm
            | StrOutputParser()
    ).with_config(run_name="TOC-fix")

    # The final full chain
    map_reduce = (map_as_doc_chain.map() | collapse | reduce_chain | toc_chain).with_config(
        run_name="Map reduce"
    )

    return map_reduce.invoke(splits, config={"max_concurrency": 30})