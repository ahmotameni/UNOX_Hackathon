from langchain.prompts import PromptTemplate


qa_template = """You have been provided with a question, delimited by '#####'. The following pieces of context which is from a csv file, delimited by '-----', will assist you in answering the question.
Your tasks are to:
1- Provide a helpful answer to the question using ONLY the information provided in the context.
2- List the source file or files that you utilized to answer the question. If you cannot answer the question with the provided context, return an empty array.
3- Respond a JSON object formatted according to the following schema:

```JSON
{{
    "answer": "string", // Answer to the question using the provided context
    "source": ["string"], // Array containing strings, each representing a source file used to craft the answer
}}
```
#####{question}#####

-----{context}-----

REMEMBER, if you don't know the answer, state that you don't know. Do not attempt to fabricate an answer.
REMEMBER, if you don't know the answer, the source should be an empty array.
REMEMBER, .Ensure that you only output the JSON, without any additional notes or explanations.
Ensure that the answer within the json is concise and informative, offering sufficient detail without being overly terse or telegraphic.
"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(qa_template)
