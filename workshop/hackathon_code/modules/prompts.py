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
initial_prompt = """
    You have been provided with a user query, delimited by '#####', I need you to tell me which keywords from the following list is related to the the query.
    Just Answer with the keywords in the list below you can find the help for each keywords in the the part delimited by *****
    ---DO NOT INTERPRET THE QUERY, JUST TELL ME WHICH KEYWORDS RELATED TO THE QUERY---
    ---DO NOT INTERPRET THE QUERY, IF YOU CANT FIND ANY KEYWORDS OUT OF FOLLOWING LISTS PUT AN EMPTY LIST IN THE OUTPUT. FOR EXAMPLE: []---

    NUMERICAL_KEYWORDS = [price, type, panel, powersupply, width, depth, height, weight, trays, traysize, distancebetweentrays, voltage, electricpower, frequency, consumption]
    CATEGORICAL_KEYWORDS = [DRY_Maxi, STEAM_Maxi22INsec, AIR_Maxi, ADAPTIVE_Cooking, AIR_Plus, SMART_Preheating, Dry_Maxi45_m3_h, STEAM_Plus, AUTO_Soft, DRY_Plus, Air_Maxi250_km_h, DDC_Ai, STEAM_Maxi, DDC_App, STEAM_Maxi14_l_sec, DDC_Unox_com,CLIMALUX, AUTO_Matic, DDC_Stats, SENSE_Klean, EFFICENt_Power, analog_control, high_productivity, professional_use, hot_fridge, automatic_cycles, internet_connection, baking_frozen_products, designed_commercial, compact_oven, self_cleaning]

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
    AIR_Maxi: Maximizes airflow within the oven to ensure even cooking.
    AIR_Plus: Enhanced air circulation technology for better heat distribution.
    AUTO_Matic: Fully automatic cooking processes for ease of use.
    AUTO_Soft: Softens automatic adjustments for delicate cooking.
    Air_Maxi250_km_h: Powerful air circulation at 250 kilometers per hour for rapid cooking.
    CLIMALUX: Advanced climate control within the oven for precise temperature management.
    DDC_Ai: Artificial intelligence-driven dynamic data control for smarter cooking.
    DDC_App: An application to monitor and control your oven remotely.
    DDC_Stats: Statistical analysis of oven usage and performance data.
    DDC_Unox_com: Online platform integration for oven management.
    DRY_Maxi: Maximum drying efficiency for dehydrating foods.
    DRY_Plus: Additional drying features for enhanced moisture removal.
    Dry_Maxi45_m3_h: High-capacity drying up to 45 cubic meters per hour.
    EFFICENT_Power: Energy-efficient power management system.
    SENSE_Klean: Sensory technology that detects when cleaning is needed.
    SMART_Preheating: Intelligent preheating system that learns and adapts to your usage patterns.
    STEAM_Maxi: Maximizes steam production for moisture-rich cooking.
    STEAM_Maxi14_l_sec: Delivers steam at a rate of 14 liters per second.
    STEAM_Maxi22_l_sec: Delivers steam at a rate of 22 liters per second for heavy-duty usage.
    STEAM_Plus: Enhanced steam features for additional cooking techniques.
    *****
    REMEMBER: The answer should not be anything but a list of keywords., just make it a list of strings like the following example: ['keyword1', 'keyword2']
    REMEMBER: DO NOT PUT ANY FEATURE NAME OTHER THAN THE ONES IN THE LISTS ABOVE : NUMERICAL_KEYWORDS, CATEGORICAL_KEYWORDS

    #####{query}#####

"""

initial_prompt = PromptTemplate.from_template(initial_prompt)

####################
# Filter generator prompt
filter_chain_prompt = """
    * DONT FORGET THAT FOR THE OUTPUT SHOULD BE JUST A JSON STRING, "filters" as the KEY AND list of columns as they value.
    * THE NUMBER OF THE FILTERS SHOULD BE EQUAL TO THE NUMBER OF THE COLUMNS
    you are given a list of columns , delimited by '#####' and a query DELIMITED BY '&&&&&', you have to give me filtered columns according to the query provided in the JSON String format with a key 'filters' and a value which is a list of column names.

    for example if the query is "Which models weigh less than 50kg and can be used for pizza and price with less than 1000 and with the voltage of 1200 and width of 15mm and 3 trays of size 15?" 
    and the columns are: 
    ['weight', 'price', 'voltage', 'width', 'trays', 'traysize', 'DRY_Maxi'] THEN THE OUTPUT SHOULD BE a JSON string LIKE the following
    :{{"filters": ["weight<50", "price<1000", "voltage<=1200", "width<=15", "trays=3", "traysize<=15", "DRY_Maxi==True"]}}

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
    AIR_Maxi: Maximizes airflow within the oven to ensure even cooking.
    AIR_Plus: Enhanced air circulation technology for better heat distribution.
    AUTO_Matic: Fully automatic cooking processes for ease of use.
    AUTO_Soft: Softens automatic adjustments for delicate cooking.
    Air_Maxi250_km_h: Powerful air circulation at 250 kilometers per hour for rapid cooking.
    CLIMALUX: Advanced climate control within the oven for precise temperature management.
    DDC_Ai: Artificial intelligence-driven dynamic data control for smarter cooking.
    DDC_App: An application to monitor and control your oven remotely.
    DDC_Stats: Statistical analysis of oven usage and performance data.
    DDC_Unox_com: Online platform integration for oven management.
    DRY_Maxi: Maximum drying efficiency for dehydrating foods.
    DRY_Plus: Additional drying features for enhanced moisture removal.
    Dry_Maxi45_m3_h: High-capacity drying up to 45 cubic meters per hour.
    EFFICENT_Power: Energy-efficient power management system.
    SENSE_Klean: Sensory technology that detects when cleaning is needed.
    SMART_Preheating: Intelligent preheating system that learns and adapts to your usage patterns.
    STEAM_Maxi: Maximizes steam production for moisture-rich cooking.
    STEAM_Maxi14_l_sec: Delivers steam at a rate of 14 liters per second.
    STEAM_Maxi22_l_sec: Delivers steam at a rate of 22 liters per second for heavy-duty usage.
    STEAM_Plus: Enhanced steam features for additional cooking techniques.
    *****

    ---DO NOT INTERPRET THE QUERY, JUST GIVE ME A JSON string, filters as the KEY AND list of columns as they value JUST A LIST OF FILTERS ACCORDING TO THE ABOVE EXAMPLE---
    ---DO NOT INTERPRET THE QUERY, IF YOU CANT FIND ANY FILTERS OUT OF FOLLOWING COLUMNS PUT AN EMPTY LIST IN THE OUTPUT. FOR EXAMPLE: {{"filters": []}}
    ---FOR EACH ONE OF THE COLUMNS YOU SHOULD PUT A FILTER IN THE COLUMNS LIST, IF YOU CANT FIND ANY FILTERS OUT OF the columns just skip the column---
    #####{columns}#####
    &&&&&{query}&&&&&
"""

filter_chain_prompt = PromptTemplate.from_template(filter_chain_prompt)
filter = ResponseSchema(name="filters",
                                       description="List of filters to be applied to the dataframe.")
# language_sa_q = ResponseSchema(name="standalone_language",
                                    # description="Language of the rephrased question.")

filter_schemas = [filter]

filter_parser = StructuredOutputParser.from_response_schemas(filter_schemas)


####################
# Code generator prompt
code_generation_prompt = """
    You have been provided with couple of filters by a list delimited with #####, give me a python code which uses pandas library so that I can filter the dataframe with the provided filters.
    Filters are delimited with -----.
    
    for example if the filters are: ["weight<50", "price<1000", "voltage<=1200", "width<=15", "trays=3", "traysize<=15", "DRY.Maxi==True"] THEN THE OUTPUT SHOULD BE a python code LIKE the following:
        => df = df[(df["weight"] < 50) & (df["price"] < 1000) & (df["width"] <= 15) & (df["trays"] == 3) & (df["DRY.Maxi'] == True)]
    
    #####{columns}#####
    -----{filters}-----
    
    - DO NOT INTERPRET THE QUERY, JUST GIVE ME A PYTHON CODE TO FILTER THE DATAFRAME, JUST A PYTHON CODE TO FILTER THE DATAFRAME ACCORDING TO THE ABOVE EXAMPLE.
    - DO NOT GIVE A CODE WITHOUT FILTERING THE COLUMNS.
"""

CODE_GEN_PROMPT = PromptTemplate.from_template(code_generation_prompt)

####################
# Answer chain
answer_chain_prompt = """
You have been provided with part of a dataframe delimited by #####. I need you to to start your sentence with "The best product for you is" and then give me the name of the product and the according url.
You need to answer politely and informatively. We need to redirect customers to the product page.
You can also add some of the product infos like price and energy consumption if you want.

#####{context}#####
"""

answer_chain_prompt = PromptTemplate.from_template(answer_chain_prompt)