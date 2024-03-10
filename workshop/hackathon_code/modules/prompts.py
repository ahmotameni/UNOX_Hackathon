from langchain.chains import LLMChain
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate

from modules.output_parsers import ResponseSchema, StructuredOutputParser

qa_template = """You are a member of UNOX Customer Service team. Your task is to help customers to find the product that suits them more among all the products available.
You have been provided with a user query, delimited by '#####'. The following pieces of context which is from a csv file, delimited by '-----', contains data of UNOX products. 

Your tasks are to:
1- You have recommend the best product(s) according to user's query using ONLY the information provided in the context.
2- If you cannot answer the question with the provided context, return an empty array.
3- Respond a JSON object formatted according to the following schema:


#####{query}#####

-----{context}-----

"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(qa_template)


# Map-Reduce prompts
########################################################################
# MAP REDUCE
map_prompt="""
You are a member of UNOX Customer Service team. Your task is to help customers to find the product that suits them more among all the products available.
You have been provided with a user query, delimited by '#####'. The following pieces of context which is from a csv file, delimited by '-----', contains data of UNOX products. 

Your tasks are to:
1- You have recommend the best product(s) according to user's query using ONLY the information provided in the context.
2- If you cannot answer the question with the provided context, return an empty array.
3- Check all the data in context section. There might be multiple products.
4- Double check that the products you find have the features that is asked in user's query.
5- In the output, JUST include the whole related row in the same way as input. Do not write anything else.

-----{context}-----

#####
"""

reduce_prompt="""
You are a member of UNOX Customer Service team. Your task is to help customers to find the product that suits them more among all the products available.
You have been provided with a user query, delimited by '#####'. The following pieces of context which is from a csv file, delimited by '-----', contains data of UNOX products. 

Your tasks are to:
1- You have recommend the best product(s) according to user's query using ONLY the information provided in the context.
2- If you cannot answer the question with the provided context, return an empty array.
3- Check all the data in context section. There might be multiple products.
4- Double check that the products you find have the features that is asked in user's query.
5- In the output, include the whole related row in the same way as input.

-----{context}-----

#####
"""

# MAP_PROMPT = PromptTemplate.from_template(map_prompt)
# REDUCE_PROMPT = PromptTemplate.from_template(reduce_prompt)

##################
# Stand alone question generator
standalone_question_template = """Given the conversation (delimited by "#####") and the follow-up question (delimited by "-----"),your objective is to rephrase the follow up question to be a standalone question.
To reach the objective

1- Rephrase the follow-up question to be a standalone question.
2- Respond in JSON format with following scheme:
Respond using a markdown code snippet with a JSON object formatted according to the following schema:

```json
{{
    "standalone_question": "string", // Rephrased question considering the chat history
}}
```
Chat History:
#####
{chat_history}
#####

Follow-up question: 
----
{question}
-----

REMEMBER, the standalone question should not be ambiguous. Whenever necessary and possible, replace ambiguous pronouns such as 'he', 'she', 'it', etc., with the specific entity’s name from the conversation history.
REMEMBER, if the chat history is empty, return the follow-up question without any modifications.
REMEMBER, do NOT add any information to the question that is not mentioned in the chat history. Rephrase the follow-up question using ONLY the information provided in the history and the follow-up question.
"""

CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_template(standalone_question_template)

standalone_q = ResponseSchema(name="standalone_question",
                                       description="Rephrased question that is clear and stand-alone.")
# language_sa_q = ResponseSchema(name="standalone_language",
                                    # description="Language of the rephrased question.")

q_generator_schemas = [standalone_q]

q_generator_parser = StructuredOutputParser.from_response_schemas(q_generator_schemas)

######################
# Initial chain

initial_chain_template = """
You have been provided with a user query, delimited by '#####', I need you to tell me which keywords from the following list is related to the the query.
    Just Answer with the keywords in the list below you can find the help for each keywords in the the part delimited by *****
    ---DO NOT INTERPRET THE QUERY, JUST TELL ME WHICH KEYWORDS RELATED TO THE QUERY---
    ---DO NOT INTERPRET THE QUERY, IF YOU CANT FIND ANY KEYWORDS OUT OF FOLLOWING LISTS PUT AN EMPTY LIST IN THE OUTPUT. FOR EXAMPLE: []---
    
    NUMERICAL_KEYWORDS = [price, type, panel, powersupply, width, depth, height, weight, trays, traysize, distancebetweentrays, voltage, electricpower, frequency, consumption]
    CATEGORICAL_KEYWORDS = [DRY.Maxi, STEAM.Maxi22 l/sec, AIR.Maxi, ADAPTIVE.Cooking, AIR.Plus, SMART.Preheating, Dry.Maxi45 m3/h, STEAM.Plus, AUTO.Soft, DRY.Plus, Air.Maxi250 km/h, DDC.Ai, STEAM.Maxi, DDC.App, STEAM.Maxi14 l/sec, DDC.Unox.com  CLIMALUX, AUTO.Matic, DDC.Stats, SENSE.Klean, EFFICENT.Power, analog_control, high_productivity, professional_use, hot_fridge, automatic_cycles, internet_connection, baking_frozen_products, designed_commercial, compact_oven, self_cleaning]


    in the end I want the answer to be in a list like the following example: ['keyword1', 'keyword2']


    *****
    price: The price of the oven.
    panel: The type of control panel the oven has.
    powersupply: The power supply of the oven.
    analog_control: Indicates if the oven has analog control.
    high_productivity: Reflects if the oven is described as highly productive.
    professional_use: True if the oven is aimed at professional use.
    hot_fridge: Whether the oven has the Hot Fridge feature to preserve food at service temperature.
    automatic_cycles: If the oven has automatic cooking cycles.
    internet_connection: Indicates internet connectivity.
    baking_frozen_products: True if the oven is suitable for baking frozen bakery products.
    designed_commercial: If the oven is specifically designed for commercial spaces.
    compact_oven: Whether the oven is compact and meant for simple confectionery/bakery processes.
    self_cleaning: If the oven is self-cleaning.
    ADAPTIVE.Cooking: Automatically adjusts cooking parameters for optimal results.
    AIR.Maxi: Maximizes airflow within the oven to ensure even cooking.
    AIR.Plus: Enhanced air circulation technology for better heat distribution.
    AUTO.Matic: Fully automatic cooking processes for ease of use.
    AUTO.Soft: Softens automatic adjustments for delicate cooking.
    Air.Maxi250 km/h: Powerful air circulation at 250 kilometers per hour for rapid cooking.
    CLIMALUX: Advanced climate control within the oven for precise temperature management.
    DDC.Ai: Artificial intelligence-driven dynamic data control for smarter cooking.
    DDC.App: An application to monitor and control your oven remotely.
    DDC.Stats: Statistical analysis of oven usage and performance data.
    DDC.Unox.com: Online platform integration for oven management.
    DRY.Maxi: Maximum drying efficiency for dehydrating foods.
    DRY.Plus: Additional drying features for enhanced moisture removal.
    Dry.Maxi45 m3/h: High-capacity drying up to 45 cubic meters per hour.
    EFFICENT.Power: Energy-efficient power management system.
    SENSE.Klean: Sensory technology that detects when cleaning is needed.
    SMART.Preheating: Intelligent preheating system that learns and adapts to your usage patterns.
    STEAM.Maxi: Maximizes steam production for moisture-rich cooking.
    STEAM.Maxi14 l/sec: Delivers steam at a rate of 14 liters per second.
    STEAM.Maxi22 l/sec: Delivers steam at a rate of 22 liters per second for heavy-duty usage.
    STEAM.Plus: Enhanced steam features for additional cooking techniques.
    *****
    
    REMEMBER: The answer should not be anything but a list of keywords., just make it a list of strings like the following example: ['keyword1', 'keyword2']
    REMEMBER: DO NOT PUT ANY FEATURE NAME OTHER THAN THE ONES IN THE LISTS ABOVE : NUMERICAL_KEYWORDS, CATEGORICAL_KEYWORDS

    #####{query}#####
"""

INITIAL_CHAIN_PROMPT = PromptTemplate.from_template(initial_chain_template)


####################
# Filter generator prompt
filter_gen_prompt = """
you are given a list of columns , delimited by '#####' and a query DELIMITED BY '&&&&&', you have to give me filtered columns according to the query provides.

    for example if the query is "Which models weigh less than 50kg and can be used for pizza and price with less than 1000 and with the voltage of 1200 and width of 15mm and 3 trays of size 15?" 
    and the columns are: 
    ['weight', 'price', 'voltage', 'width', 'trays', 'traysize', 'DRY.Maxi'] THEN THE OUTPUT SHOULD BE:['weight<50', 'price<1000', 'voltage<=1200', 'width<=15', 'trays=3', 'traysize<=15', 'DRY.Maxi==True']

    ---DO NOT INTERPRET THE QUERY, JUST GIVE ME A LIST JUST A LIST OF FILTERS ACCORDING TO THE ABOVE EXAMPLE---
    ---DO NOT INTERPRET THE QUERY, IF YOU CANT FIND ANY FILTERS OUT OF FOLLOWING COLUMNS PUT AN EMPTY LIST IN THE OUTPUT. FOR EXAMPLE: []---
    #####{columns}#####
    &&&&&{query}&&&&&
"""

FILTER_GEN_PROMPT = PromptTemplate.from_template(filter_gen_prompt)


