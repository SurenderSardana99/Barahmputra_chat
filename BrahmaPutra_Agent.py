import os
import traceback
import re
import pinecone
import numpy as np
import guidance
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import AzureOpenAI
import openai
# import nest_asyncio
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import tiktoken
import time
from time import ctime
from datetime import datetime, timedelta
import ntplib
import pytz
import json
from sqlalchemy import create_engine
from sqlalchemy import inspect
import pandas as pd
import mysql.connector
import pdb
import ast
from messages import (
    driving_message,
    inquiry_driving_message,
    messagesForExpandSearch,
    messagesForMultipleResults,
    messagesForResult,
    messagesForResultAccepted,
    messagesForResultDenied,
    messagesForResultDeniedFinal,
    messagesForInquiry,
    messageForRefinementPossible,
    messageForCategoricalRefinement
)
from messages import (
    non_inquiry_embedding_1, non_inquiry_embedding_2, non_inquiry_embedding_3, non_inquiry_embedding_4, non_inquiry_embedding_5, non_inquiry_embedding_6, non_inquiry_embedding_7, non_inquiry_embedding_8, non_inquiry_embedding_9, non_inquiry_embedding_10,
    inquiry_embedding_1, inquiry_embedding_2, inquiry_embedding_3, inquiry_embedding_4, inquiry_embedding_5, inquiry_embedding_6, inquiry_embedding_7, inquiry_embedding_8, inquiry_embedding_9
)
### Test
##### Section 1: SQL Functions ######
def save_input_for_rl(text, chat_stage):
    """
    Code to save everyone's input into a DB to be used for reinforcement learning or training a custom classifier.
    """
    #Initialize sql connection
    mydb = mysql.connector.connect(
        host = "20.212.32.214",
        user = "UserDataDb",
        password = "sa_54321",
        database = "datamart",
    )
    mycursor = mydb.cursor()
    sql = """INSERT INTO datamart.brahmaputrainputs 
    (input, chatStage) VALUES (%s, %s)
    """
        
    val = (text, chat_stage)
    mycursor.execute(sql, val)
    mydb.commit()
    mydb.close()
    print(mycursor.rowcount, "record inserted.")

def get_table_from_sql(session_id):
    """
    Get a row from sql. Used to read variables
    """
    #Initialize sql connection
    host = "20.212.32.214"
    user = "UserDataDb"
    password = "sa_54321"
    database = "datamart"
    connection = mysql.connector.connect(host=host,
                                         database=database,
                                         user=user,
                                         password=password)
    query_df = f"SELECT * FROM datamart.BrahmaputraTableDev WHERE session_id = '{session_id}';"
    print("Query_df", query_df)
    cursor = connection.cursor()
    cursor.execute(query_df)
    # get all records
    records = cursor.fetchall()
    df = pd.DataFrame.from_records(records, columns=[x[0] for x in cursor.description])
    connection.close()
    return df

def save_table_into_sql(val1=None, val2=None, val3=None, val4='driving', val5=None, val6=None, val7=None, val8=None, val9=None, val10=None, val11=None, val12=None):
    """
    Save an extra row into the DB
    """
    mydb = mysql.connector.connect(
        host = "20.212.32.214",
        user = "UserDataDb",
        password = "sa_54321",
        database = "datamart",
    )
    mycursor = mydb.cursor()
    
    sql = """INSERT INTO datamart.BrahmaputraTableDev 
    (TableName1, TableName2, TableName3, ChatStage, previous_intent, session_id, table_choice, table_choice_desc, TableDesc1, TableDesc2, TableDesc3, refine_dimension ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
        
    val = (val1, val2, val3, val4, val5, val6, val7, val8, val9, val10, val11, val12)
    mycursor.execute(sql, val)
    mydb.commit()
    mydb.close()
    print(mycursor.rowcount, "record inserted.")

def update_table_into_sql(val1=None, val2=None, val3=None, val4='driving', val5=None, val6=None, val7=None, val8=None, val9=None, val10=None, val11=None, val12=None):
    """
    Code to write into the DB some variables for different stages of the conversation.
    """
    #Error handling for if session id not detected
    if not val6:
        return "Error! Session ID not detected by update_table_into_sql"
    
    mydb = mysql.connector.connect(
        host = "20.212.32.214",
        user = "UserDataDb",
        password = "sa_54321",
        database = "datamart",
    )
    mycursor = mydb.cursor()

    sql = """UPDATE BrahmaputraTableDev 
    SET TableName1 = %s, TableName2 = %s, TableName3 = %s, ChatStage = %s, previous_intent = %s, table_choice = %s, table_choice_desc = %s, TableDesc1 = %s, TableDesc2 = %s, TableDesc3 = %s, refine_dimension= %s
    WHERE session_id = %s
    """
        
    val = (val1, val2, val3, val4, val5, val7, val8, val9, val10, val11, val12, val6)
    mycursor.execute(sql, val)
    mydb.commit()
    mydb.close()
    print(mycursor.rowcount, "record inserted.")

def check_session_id_exists(session_id):
    """
    Checks if the currently generated Session ID exists.
    """
    mydb = mysql.connector.connect(
        host = "20.212.32.214",
        user = "UserDataDb",
        password = "sa_54321",
        database = "datamart",
    )
    mycursor = mydb.cursor()
    
    query = f"""SELECT * FROM datamart.BrahmaputraTableDev WHERE session_id = '{session_id}'"""
    mycursor.execute(query)
    # Check if row with session_id exists
    if mycursor.fetchone():
        mydb.close()
        return True
    else:
        mydb.close()
        return False
    
def reset_conversation(session_id, driving_message=driving_message):
    """
    Resets the conversation by writing a SQL query to update the values.
    At the moment, this is the ONLY way to reset the conversation. 
    """
    update_table_into_sql(val1=None, val2=None, val3=None, val4='driving', val5=None, val6=session_id, val7=None, val8=None, val9=None, val10=None, val11=None)
    driving_message = [message for message in driving_message if message['role'] == 'system']
    return driving_message

##### Section 2: Environment Variables #####
openai.api_type = "azure"
openai.api_base = "https://metadata-openai-instance.openai.azure.com"
openai.api_version = "2023-07-01-preview"
openai.api_key = "a53834cd090f4839a38009e04a67ab50"


os.environ['OPENAI_API_TYPE']="azure"
os.environ['OPENAI_API_KEY']="a53834cd090f4839a38009e04a67ab50"
os.environ['OPENAI_API_BASE']="https://metadata-openai-instance.openai.azure.com"
os.environ['OPENAI_API_VERSION']="2022-12-01"

embeddings = OpenAIEmbeddings(deployment="text-embedding-ada-002")

##### Section 3: Variables #####
saved_table_name=""
saved_table_desc=""
saved_input=""

yes_embedding = embeddings.embed_query("yes, proceed")
no_embedding = embeddings.embed_query("no, do not proceed")

##### Section 4: Misc Functions #####
def reload_driving_message():
    return([message for message in driving_message if message['role'] == 'system'])

def intent_classifier(input, intents):
    guidance.llm = guidance.llms.OpenAI(
            'text-davinci-003',
            api_type=os.environ["OPENAI_API_TYPE"],
            api_key=os.environ["OPENAI_API_KEY"],
            api_base=os.environ["OPENAI_API_BASE"],
            api_version=os.environ["OPENAI_API_VERSION"],
            deployment_id='text-davinci-003',
            caching=True,
        )

    program = guidance("""
    Given the user input: {{input}}
    Select one of the following intents: {{intents}}
    
    ```json
    {
        "input": "{{input}}",
        "options": "{{intents}}",
        "intent":"{{gen 'answer' stop='"'}}"
    }
    ```
    """)

    executed_program = program(
        input=input,
        intents=intents
    )
    print(f"Intent for {input} is {executed_program['answer']}")
    return(executed_program['answer'])

def guidance_filter_item_identifier(input, filter_items):
    guidance.llm = guidance.llms.OpenAI(
            'text-davinci-003',
            api_type=os.environ["OPENAI_API_TYPE"],
            api_key=os.environ["OPENAI_API_KEY"],
            api_base=os.environ["OPENAI_API_BASE"],
            api_version=os.environ["OPENAI_API_VERSION"],
            deployment_id='text-davinci-003',
            caching=True,
        )
    program = guidance("""
    Given the user input: {{input}}
    Select either one or many of these filter items: {{filterItems}}
    If none, set the answer to be "--None--"
                       
    ```json
    {
        "input": "{{input}}",
        "options": "{{filterItems}}",
        "answer":"{{gen 'answer' stop='"'}}"
    }
    ```
    """)

    executed_program = program(
        input=input,
        filterItems=filter_items
    )
    print(f"Intent for {input} is {executed_program['answer']}")
    return(executed_program['answer'])

def clean_response(text):
    """
    Used to clean the response in the event the LLM returns a string that starts with "response:" or "ADA:"
    TODO: See if it can be fixed with guidance
    """
    return text.replace("Response:", "").replace("response:", "").replace("ADA:", "").replace("ada:","")

def cosine_similarity(vector1, vector2):
    """
    Implementation of the cosine similarity formula to calculate semantic closeness between two phrases
    """
    # Convert the lists to NumPy arrays
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    # Calculate the dot product of the two vectors
    dot_product = np.dot(vector1, vector2)
    # Calculate the L2 norm (Euclidean norm) of each vector
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    # Calculate the cosine similarity
    similarity = dot_product / (norm1 * norm2)
    return similarity

def pinecone_table_search(userInputQuestion, client_Id, k=1, filter={}):
    """
    Runs a similarity search in pinecone to get out a certain number of tables

    Keyword arguments:
    userInputQuestion -- User Input
    k -- Number of results to return
    filter -- What kind of filters to try and add in
    """
    pinecone.init(api_key="32acb893-7a53-4a9d-8ea0-746de30755d3", environment="asia-southeast1-gcp-free")
    index_name = "userdata-metadata-development"  # Replace with your actual index name
    index = pinecone.Index(index_name=index_name)
    
    embeddings = OpenAIEmbeddings(deployment="text-embedding-ada-002")
    query_vector = embeddings.embed_query(userInputQuestion)
    print('filter is ', filter)
    print("inside pinecone_table_search , client_id=",client_Id)
    namespaceClient='table_level_metadata_namespace_demo'
    if (client_Id=="mag001"):
        namespaceClient="mag_001_001_das_tbl_metadata_clnt"
    elif (client_Id=="starhub_001"):
        namespaceClient="sth_001_001_das_tbl_metadata"
  
    print("namespaceClient" , namespaceClient)
    
    resultsForTable = index.query(
            vector=query_vector,
            top_k=k,
            include_values=True,
            include_metadata=True,
            namespace = namespaceClient,
            filter = filter
            )

    # print("Score:" ,resultsForTable['matches'][0]['score'])
    # print("Id:" ,resultsForTable['matches'][0]['id'])
    # print("Table Description:" ,resultsForTable['matches'][0]['metadata']['table_description'])
    return (resultsForTable['matches'])

def pinecone_column_search(client_Id, userInputQuestion, k=1, filter={}):
    """
    Runs a similarity search in pinecone to get out a certain number of tables

    Keyword arguments:
    userInputQuestion -- User Input
    k -- Number of results to return
    filter -- What kind of filters to try and add in
    """
    pinecone.init(api_key="32acb893-7a53-4a9d-8ea0-746de30755d3", environment="asia-southeast1-gcp-free")
    index_name = "userdata-metadata-development"  # Replace with your actual index name
    index = pinecone.Index(index_name=index_name)

    embeddings = OpenAIEmbeddings(deployment="text-embedding-ada-002")
    query_vector = embeddings.embed_query(userInputQuestion)
    column_level_Namespace="column_level_metadata_namespace_demo"
    if(client_Id=="mag001"):
        column_level_Namespace="mag_001_001_das_col_metadata_clnt"
    elif (client_Id=="starhub_001"):
        column_level_Namespace="sth_001_001_das_col_metadata"
    resultsForColumn = index.query(
            vector=query_vector,
            top_k=1,
            include_metadata=True,
            include_values=False,
            namespace = column_level_Namespace,
            filter = filter
            )
    print("Score:" ,resultsForColumn['matches'][0]['score'])
    print("Id:" ,resultsForColumn['matches'][0]['id'])
    #print("Table Description:" ,resultsForColumn['matches'][0]['metadata']['long_description'])
    return resultsForColumn

def pinecone_table_list_search(client_Id):
    pinecone.init(api_key="32acb893-7a53-4a9d-8ea0-746de30755d3", environment="asia-southeast1-gcp-free")
    index_name = "userdata-metadata-development"  # Replace with your actual index name
    index = pinecone.Index(index_name=index_name)
    print("inside pinecone_table_list_search , client_id=",client_Id)
    namespace='table_level_metadata_namespace_demo'
    if(client_Id=="mag001"):
        namespace="mag_001_001_das_tbl_metadata_clnt"
    elif(client_Id=="starhub_001"):
        namespace="sth_001_001_das_tbl_metadata"
    stats = index.describe_index_stats()
    results = []
    res = index.query(
        vector=[0] * 1536, 
        top_k=1000, 
        include_metadata=True,
        include_values=False,
        namespace=namespace)
    for match in res['matches']:
        results.append(match['id'])
    return (results)

def extract_results(resultsForTable):
    """
    Extracts results from a Pinecone output. To be used in Pinecone related outputs
    """
    return (resultsForTable[0]['score'],
            resultsForTable[0]['id'],
            resultsForTable[0]['metadata']['table_description'])

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """
    Taken from Langchain. To avoid going over the token limit.
    """
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def process_input(input_value):
    try:
        # Try to evaluate the input value as a literal expression using ast.literal_eval
        evaluated_value = ast.literal_eval(input_value)
        
        # Check if the evaluated value is a list
        if isinstance(evaluated_value, list):
            # If it's a list, return the list
            return evaluated_value
        else:
            # If it's not a list, return the input as a string
            return input_value
    except (ValueError, SyntaxError):
        # If ast.literal_eval fails (e.g., input is a plain string), return the input as a string
        return input_value
    
##### Section 5: Main BrahmaPutra functions #####
def get_driving_message(input,driving_message=driving_message):
        # Intent Not Found
    driving_message.append({"role":"user", "content": input})
    chat = openai.ChatCompletion.create(
        engine="gpt-35-turbo",
        messages=driving_message
    )
    reply_driving_message = chat.choices[0].message.content
    driving_message.append({"role":"assistant", "content": reply_driving_message})
    print("Driving message")
    print(driving_message)
    return reply_driving_message

def get_inquiry_driving_message(input, data_dict, inquiry_driving_message=inquiry_driving_message):
        # Intent Not Found
    inquiry_driving_message.append({"role":"user", "content": f"Relevant table is found: {data_dict.loc[0,'table_choice_desc']}"})
    inquiry_driving_message.append({"role":"user", "content": input})
    chat = openai.ChatCompletion.create(
        engine="gpt-35-turbo",
        messages=inquiry_driving_message
    )
    reply_driving_message = chat.choices[0].message.content
    inquiry_driving_message.append({"role":"assistant", "content": reply_driving_message})
    print("Driving message")
    print(inquiry_driving_message)
    return reply_driving_message

def get_results_for_table(client_Id, data_dict, input, filter=None):
    if data_dict.loc[0, "TableName1"] is not None:
        print("Return Saved Table Detected, getting 3 results")
        results_table = pinecone_table_search(input, client_Id, k=3, filter={"table_description": {"$ne":f"{data_dict.loc[0, 'TableName1']}"}})
        intent_db_sim_score = []
        table_name = []
        table_desc = []
        for i in results_table:
            temp_intent_db_sim_score,temp_table_name,temp_table_desc = extract_results([i])
            print("Table found (name): ", temp_table_name)
            print("Score:", temp_intent_db_sim_score)
            print("Table found (description): ", temp_table_desc)
            intent_db_sim_score.append(temp_intent_db_sim_score)
            table_name.append(temp_table_name)
            table_desc.append(temp_table_desc)
        return (intent_db_sim_score, table_name, table_desc)

    else:
        print("Return saved table not detected, getting one result")
        results_table = pinecone_table_search(input, client_Id)
        intent_db_sim_score, table_name, table_desc = extract_results(results_table)
        return (intent_db_sim_score, table_name, table_desc)

def results_table_found(input, intent_db_sim_score, table_name, table_desc, session_id, messagesForResult=messagesForResult):
    if isinstance(intent_db_sim_score, float): #First round, only looking for 1 answer
        messagesForResult.append({"role":"user","content":f"Relevant Table is Found: {table_desc}. Be sure to follow the format provided"})
        chat = openai.ChatCompletion.create(
            engine="gpt-35-turbo",
            messages=messagesForResult
        )
        reply_table_found=chat.choices[0].message.content
        print(messagesForResult)
    
        # saved_table_name = table_name
        update_table_into_sql(val1=table_name, val4='confirming_single', val5=input, val6=session_id, val9=table_desc)
        saved_table_desc = table_desc
        saved_input = input
    
        # returned_saved_table_name = saved_table_name
        # previous_table_desc = table_desc
        print("giving user the choice of 1 table")
        return (reply_table_found, saved_table_desc, saved_input)
    else: #Second round, looking for 3 answers
        messagesForMultipleResults.append({"role":"user","content":f"Relevant Tables are Found: {table_desc}. Be sure to follow the format provided"})
        chat = openai.ChatCompletion.create(
            engine="gpt-35-turbo",
            messages=messagesForMultipleResults
        )
        reply_table_found=chat.choices[0].message.content
        ### HARDCODED FOR NOW, MUST REFACTOR WHEN SCALING
        try:
            val1 = table_name[0]
            val9 = table_desc[0]
        except:
            print("table_name[0] not found")
            val1 = None
            val9 = None
        try:
            val2 = table_name[1]
            val10 = table_desc[1]
        except:
            print("table_name[1] not found")
            val2 = None
            val10 = None
        try:
            val3 = table_name[2]
            val11 = table_desc[2]
        except:
            print("table_name[2] not found")
            val3 = None    
            val11 = None

        update_table_into_sql(val1=val1, val2=val2, val3=val3, val4='confirming_multiple', val5=input, val6=session_id, val9=val9, val10=val10, val11=val11)
        print("giving user the choice of 3 table")
        saved_table_desc = table_desc
        saved_input = input
        return (reply_table_found, saved_table_desc, saved_input)

def table_dimensions_found(client_Id, input, messagesForRefinementPossible=messageForRefinementPossible):
    pinecone.init(api_key="32acb893-7a53-4a9d-8ea0-746de30755d3", environment="asia-southeast1-gcp-free")
    index_name = "userdata-metadata-development"  # Replace with your actual index name
    index = pinecone.Index(index_name=index_name)
    embeddings = OpenAIEmbeddings(deployment="text-embedding-ada-002")
    query_vector = embeddings.embed_query(input)
    column_level_Namespace="column_level_metadata_namespace_demo_v2"
    if(client_Id=="mag001"):
        column_level_Namespace="mag_001_001_das_col_metadata_clnt"
    if(client_Id=="starhub_001"):
        column_level_Namespace="sth_001_001_das_col_metadata"
    resultsForTable = index.query(
            vector=query_vector,
            top_k=10000,
            include_values=False,
            include_metadata=True,
            namespace = column_level_Namespace,
            filter={"sql_path": {"$eq":input}}
            )
    available_column_dimensions = []
    for i in resultsForTable['matches']:
        if (i['metadata']['dimension'] == 'Yes') and (i['metadata']['data_tribe'] in ['numerical', 'categorical', 'date related']):
            available_column_dimensions.append(f"column_name:{i['metadata']['cleaned_column_name']}, column_description:{i['metadata']['column_description']}")
    messagesForRefinementPossible = [message for message in messagesForRefinementPossible if message['role'] != 'user']
    messagesForRefinementPossible.append({"role":"user", "content":f"Column dimensions available: {available_column_dimensions}"})
    chat = openai.ChatCompletion.create(
        engine="gpt-35-turbo",
        messages=messagesForRefinementPossible
    )
    reply_refinement_possible=clean_response(chat.choices[0].message.content)
    print("REFINEMENT POSSIBLE:", reply_refinement_possible)
    return reply_refinement_possible

def accept_single_table_choice(data_dict, session_id, messagesForResultAccepted=messagesForResultAccepted, saved_table_desc=saved_table_desc):
    print("Choice accepted")
    messagesForResultAccepted = [message for message in messagesForResultAccepted if message['role'] != 'user']
    messagesForResultAccepted.append({"role":"user", "content":f"Table to analyze: {saved_table_desc}"})
    chat = openai.ChatCompletion.create(
        engine="gpt-35-turbo",
        messages=messagesForResultAccepted
    )
    reply_result_accepted=clean_response(chat.choices[0].message.content)
    
    temp_saved_table_name=data_dict.loc[0,"TableName1"]
    temp_saved_table_desc=data_dict.loc[0,"TableDesc1"]
    returned_saved_input=data_dict.loc[0,"previous_intent"]
    saved_table_name=""
    update_table_into_sql(val1=None, val4='inquiry', val6=session_id, val7=temp_saved_table_name, val8=temp_saved_table_desc)
    saved_table_desc=""
    saved_input=""

    return [{'reloadSixups': True},
            temp_saved_table_name,
            returned_saved_input,
            reply_result_accepted]

def reject_single_table_choice(data_dict, session_id, messagesForResultDenied=messagesForResultDenied, messagesForResult=messagesForResult, driving_message=driving_message):
    print("Message denied")
    messagesForResultDenied = [message for message in messagesForResultDenied if message['role'] != 'user']
    print(messagesForResultDenied)
    chat = openai.ChatCompletion.create(
        engine="gpt-35-turbo",
        messages=messagesForResultDenied
    )
    reply_result_denied=chat.choices[0].message.content

    messagesForResult = [message for message in messagesForResult if message['role'] != 'user']
    driving_message = [message for message in driving_message if message['role'] == 'system']
    
    saved_table_desc=""
    saved_input=""
    update_table_into_sql(val1=data_dict.loc[0,"TableName1"], val4='driving', val6=session_id)
    driving_message.append({"role":"user", "content": input})
    driving_message.append({"role":"assistant", "content": reply_result_denied})
    returned_saved_input=saved_input

    return reply_result_denied

def multiple_table_choice(input, data_dict, session_id, client_Id, messagesForResultDeniedFinal=messagesForResultDeniedFinal, messagesForResultAccepted=messagesForResultAccepted, messagesForResult=messagesForResult, driving_message=driving_message):
    intent_db_sim_score, table_name, table_desc = get_results_for_table(client_Id=client_Id, data_dict=data_dict, input=data_dict.loc[0,'previous_intent'])
    print("saved_table_name is a list, choosing 1 of 3")
    question = f'''
    I found {len(table_name)} tables that might relate to Chris's query.
    Below are the table(s) that I found:
    {table_desc}

    Below is Chris's response towards my finding:
    {input}

    Keep in mind that Chris might answer according to the table's name or table's number/sequence. Which table does Chris wants? 

    Sample of Expected Answer 1:
    {{"chris_wants":"table 1"}}
    Sample of Expected Answer 2:
    {{"chris_wants":"table 2"}}
    Sample of Expected Answer 3:
    {{"chris_wants":"table 3"}}
    Sample of Expected Answer 4:
    {{"chris_wants":"None"}}

    !!!IMPORTANT!!! DO NOT GIVE ANSWER OUTSIDE THE 4 SAMPLE OF EXPECTED ANSWER!!!
    Answer (Please give it in a dictionary format like the sample of expected answers above):
    '''
    llm = AzureOpenAI(
        temperature=0.1,
        verbose=True,
        deployment_name='gpt-35-turbo',
        model_name='gpt-35-turbo',
        max_tokens=3900-num_tokens_from_string(question,'gpt-3.5-turbo'))

    response=llm.predict(question)
    index_of_brace = response.find('}')

    if index_of_brace != -1:
        response = response[:index_of_brace + 1]
    else:
        response = response
    eval_message = eval(response)
    print("EVAL MESSAGE",eval_message)
    if eval_message['chris_wants'] == "None": #### If message implies none are fit
        messagesForResultDeniedFinal = [message for message in messagesForResultDeniedFinal if message['role'] != 'user']
        messagesForResultDeniedFinal.append({"role":"user", "content": input})
        chat = openai.ChatCompletion.create(
            engine="gpt-35-turbo",
            messages=messagesForResultDeniedFinal
        )   
        reply_result_denied_final=chat.choices[0].message.content
        print("Picked NONE of the 3 tables")
        driving_message_temp = reset_conversation(session_id=session_id)
        driving_message = driving_message_temp

        return [reply_result_denied_final,""]
    else: #### If message implies one of them fit
        match = int(eval_message['chris_wants'][-1])
        data_dict_filter = data_dict[f"TableDesc{str(match)}"]
        print("Messages for result accepted",messagesForResultAccepted)
        messagesForResultAccepted = [message for message in messagesForResultAccepted if message['role'] != 'user']
        messagesForResultAccepted.append({"role":"user", "content":f"Table to analyze: {data_dict_filter}"})
        chat = openai.ChatCompletion.create(
            engine="gpt-35-turbo",
            messages=messagesForResultAccepted
        )   
        reply_result_accepted=clean_response(chat.choices[0].message.content)
        print("reply_result_accepted")
        print(reply_result_accepted)

        messagesForResult = [message for message in messagesForResult if message['role'] != 'user']
        driving_message = [message for message in driving_message if message['role'] == 'system']
        
        returned_saved_input=data_dict.loc[0,"previous_intent"]
        temp_saved_table_name = data_dict.loc[0,f"TableName{match}"]
        saved_table_desc = get_table_desc(client_Id, temp_saved_table_name)
        update_table_into_sql(val1=None, val4='inquiry', val6=session_id, val7=temp_saved_table_name, val8=saved_table_desc)

        return ([{'reloadSixups': True},
                temp_saved_table_name,
                returned_saved_input,
                reply_result_accepted],saved_table_desc)

def inquire_column(data_dict, input, client_Id, messagesForInquiry=messagesForInquiry):
    test_input_embedding = embeddings.embed_query(input)
    list_of_inquiry_embeddings=[inquiry_embedding_1,inquiry_embedding_2,inquiry_embedding_3,inquiry_embedding_4,inquiry_embedding_5,inquiry_embedding_6,inquiry_embedding_7,inquiry_embedding_8,inquiry_embedding_9]
    list_of_non_inquiry_embeddings=[non_inquiry_embedding_1,non_inquiry_embedding_2,non_inquiry_embedding_3,non_inquiry_embedding_4,non_inquiry_embedding_5,non_inquiry_embedding_6,non_inquiry_embedding_7,non_inquiry_embedding_8,non_inquiry_embedding_9,non_inquiry_embedding_10]
    similarity_inquiry = max([cosine_similarity(test_input_embedding, inquiry_embedding) for inquiry_embedding in list_of_inquiry_embeddings])
    similarity_non_inquiry = max([cosine_similarity(test_input_embedding, non_inquiry_embedding) for non_inquiry_embedding in list_of_non_inquiry_embeddings])
    if similarity_inquiry > similarity_non_inquiry: ## If we define that it is an inquiry
        column_description = pinecone_column_search(client_Id, userInputQuestion=input ,filter={"sql_path": {"$eq":f"{data_dict.loc[0, 'table_choice']}"}})
        messagesForInquiry.append({"role":"user","content":f"Relevant Column Description is Found: {column_description}. Be sure to follow the format provided"})
        chat = openai.ChatCompletion.create(
            engine="gpt-35-turbo",
            messages=messagesForInquiry
        )
        temp_saved_table_name=data_dict.loc[0,"table_choice"]
        returned_saved_input=data_dict.loc[0,"previous_intent"]
        return [temp_saved_table_name,
            returned_saved_input,
            chat.choices[0].message.content]
    else: ## If we determine that it is NOT an inquiry
        reply_result_denied = "I'm sorry, it seems like this is not an inquiry. Feel free to inquire about any of the columns in the table. If you would like to reset the conversation, type '--reset--'"
        temp_saved_table_name=data_dict.loc[0,"table_choice"]
        returned_saved_input=data_dict.loc[0,"previous_intent"]
        return [temp_saved_table_name,
            returned_saved_input,
            reply_result_denied]

def check_for_dimension(client_Id, data_dict, input, table_choice=None):
    pinecone.init(api_key="32acb893-7a53-4a9d-8ea0-746de30755d3", environment="asia-southeast1-gcp-free")
    index_name = "userdata-metadata-development"  # Replace with your actual index name
    index = pinecone.Index(index_name=index_name)
    if not table_choice:
        table_choice = data_dict.loc[0,'table_choice']
    query_vector = [0] * 1536
    column_level_Namespace="column_level_metadata_namespace_demo_v2"
    if(client_Id=="mag001"):
        column_level_Namespace="mag_001_001_das_col_metadata_clnt"
    if(client_Id=="starhub_001"):
        column_level_Namespace="sth_001_001_das_col_metadata"
    resultsForTable = index.query(
            vector=query_vector,
            top_k=10000,
            include_values=False,
            include_metadata=True,
            namespace = column_level_Namespace,
            filter={"sql_path": {"$eq":f"{table_choice}"}}
            # filter={"sql_path": {"$eq":f"datalake.bike"}}
            )
    for result in resultsForTable['matches']:
        if result['metadata']['dimension'] == 'Yes':
            # update_table_into_sql(val4='refining', val5=input)
            return True
    return False

    
def generate_dimension_filter_and_items(client_Id, data_dict, input, messageForRefinementPossible=messageForRefinementPossible, refine_dimension=None):
    pinecone.init(api_key="32acb893-7a53-4a9d-8ea0-746de30755d3", environment="asia-southeast1-gcp-free")
    index_name = "userdata-metadata-development"  # Replace with your actual index name
    index = pinecone.Index(index_name=index_name)
    embeddings = OpenAIEmbeddings(deployment="text-embedding-ada-002")
    if refine_dimension:
        query = refine_dimension
        query_vector = embeddings.embed_query(refine_dimension)
    else:
        query = input
        query_vector = embeddings.embed_query(input)
    #breakpoint()
    column_level_Namespace="column_level_metadata_namespace_demo_v2"
    if(client_Id=="mag001"):
        column_level_Namespace="mag_001_001_das_col_metadata_clnt"
    if(client_Id=="starhub_001"):
        column_level_Namespace="sth_001_001_das_col_metadata"
    resultForColumn = index.query(
        vector=query_vector,
        top_k=1,
        include_values=False,
        include_metadata=True,
        namespace = column_level_Namespace,
        filter={"sql_path": {"$eq":f"{data_dict.loc[0,'table_choice']}"}}
    )
    # try:
    #     # Best possible result: Pinecone EXACT Search
    #     resultsForColumn = index.query(
    #             #vector=[0]*1536,
    #             vector=query_vector,
    #             top_k=10000,
    #             include_values=False,
    #             include_metadata=True,
    #             namespace = 'column_level_metadata_namespace_demo_v2',
    #             filter={"sql_path": {"$eq":f"{data_dict['table_choice'][0]}"}}
    #             )
    #     available_column_dimensions = []
    #     for i in resultsForColumn['matches']:
    #         if (i['metadata']['dimension'] == 'Yes') and (i['metadata']['data_tribe'] in ['numerical', 'categorical', 'date_related']):
    #             available_column_dimensions.append(i['metadata']['cleaned_column_name'])
                
    #     breakpoint()
    #     #TODO: USE LEVENSHTEIN DISTANCE, SET A THRESHOLD
    #     processed_input = input.strip().lower()
    #     word_match = None
    #     for word in available_column_dimensions:
    #         if word.strip().lower() == processed_input:
    #             word_match = word
    #             break
    #         else:
    #             pass

    #     # Check if the processed input matches any of the processed possible strings
    #     resultForColumn = index.query(
    #         vector=[0]*1536,
    #         top_k=1,
    #         include_values=False,
    #         include_metadata=True,
    #         namespace = 'column_level_metadata_namespace_demo_v2',
    #         filter={"cleaned_column_name": {"$eq":f"{input}"}}
    #         )
    # except:
    #     # 2nd best result: Pinecone Semantic Search 
    #     resultForColumn = index.query(
    #             vector=query_vector,
    #             top_k=1,
    #             include_values=False,
    #             include_metadata=True,
    #             namespace = 'column_level_metadata_namespace_demo_v2',
    #             filter={"sql_path": {"$eq":f"{data_dict.loc[0,'table_choice']}"}}
    #             )
    filter_name = resultForColumn['matches'][0]['id'].split(".")[-1]
    data_tribe = resultForColumn['matches'][0]['metadata']['data_tribe']
    filter_items = resultForColumn['matches'][0]['metadata']['10_samples'].split("|")
    print("DATA TRIBE:", data_tribe)
    if data_dict.loc[0,'ChatStage'] == "refining_pt1":
        if data_tribe in ['categorical', 'string', 'textual']:
            ## Categorical is actual categories
            ## String is possibly a mismatch
            ## Textual is ID
            messageForCategoricalRefinement.append({"role":"user", "content":f"I have chosen to filter by the dimension '{filter_name}'. The examples for this dimension are: {filter_items}. Be sure to follow the format provided."})
            chat = openai.ChatCompletion.create(
                engine="gpt-35-turbo",
                messages=messageForCategoricalRefinement
            )
            return (filter_name, chat.choices[0].message.content)
        elif data_tribe in ['numerical', 'date_related']:
            return (filter_name, "ADD THE INTENT CLASSIFIER FOR EXACT, MORE THAN EQ, LESS THAN EQ, BETWEEN, NOT BETWEEN")
        elif data_tribe == 'unknown':
            return (filter_name, "DATA TRIBE OF 'unknown' DETECTED. WHAT IS THIS DATA TRIBE?")
        else:
            return (filter_name, "UNRECOGNIZED DATA TRIBE, THIS SHOULD NOT HAPPEN")
    elif data_dict.loc[0,'ChatStage'] == "refining_pt2":
        if data_tribe in ['categorical', 'string', 'textual']:
            return (filter_items)
        elif data_tribe in ['numerical', 'date_related']:
            return ("PT2 REFINING NOT HANDLED YET NUMERICAL AND DATE RELATED")
        elif data_tribe == 'unknown':
            return ("PT2 REFINING DATA TRIBE UNKNOWN")
        else:
            return ("PT2 REFINING UNRECOGNIZED DATA TRIBE, THIS SHOULD NOT HAPPEN")

def get_table_desc(client_Id, table_name):
    pinecone.init(api_key="32acb893-7a53-4a9d-8ea0-746de30755d3", environment="asia-southeast1-gcp-free")
    index_name = "userdata-metadata-development"  # Replace with your actual index name
    index = pinecone.Index(index_name=index_name)

    embeddings = OpenAIEmbeddings(deployment="text-embedding-ada-002")
    query_vector = embeddings.embed_query(table_name)
    table_level_Namespace="table_level_metadata_namespace_demo"
    if(client_Id=="mag001"):
        table_level_Namespace="mag_001_001_das_tbl_metadata_clnt"
    if(client_Id=="starhub_001"):
        table_level_Namespace="sth_001_001_das_tbl_metadata"
    match = index.query(
            vector=query_vector,
            top_k=1,
            include_values=False,
            include_metadata=True,
            namespace = table_level_Namespace,
            )
    return(match['matches'][0]['metadata']['table_description'])

def remove_backslashes_and_quotes(string):
    return string.strip('"\'')

##### Section 6: Main Function #####
def chatbot(input, session_id, client_Id, user_Id):
    print("\n\n")
    print("#################STARTING NEW SESSION#################")
    print(f"#User Input:{input}")

    #Instantiate Global Variables
    global messagesForResultAccepted
    global messagesForMultipleResults
    global messagesForResultDenied
    global messagesForResultDeniedFinal
    global messagesForResult
    global messagesForExpandSearch #Added by guowei
    global driving_message
    global saved_table_name
    global saved_table_desc
    global saved_input
    global yes_embedding
    global no_embedding
    
    client_Id = remove_backslashes_and_quotes(client_Id.lower().strip())
    user_Id = remove_backslashes_and_quotes(user_Id.lower().strip())
    
    static_client_id = ["mag001", "userdata", "starhub_001"]
    print("client_Id:", client_Id)
    if client_Id not in static_client_id:
        return "Access denied. You are not authorized to access this service."
    
    ### 5.1 Create new table if session_id did not exist
    if not check_session_id_exists(session_id=session_id):
        save_table_into_sql(val6=session_id)
        driving_message = reload_driving_message()
        print(f"Create new table with session id {session_id}")
    else:
        print("Session ID already exists AND is not None")

    data_dict = get_table_from_sql(session_id) #Get from SQL the row associated with session_id
    save_input_for_rl(input, data_dict.loc[0, 'ChatStage']) #Save the input

    ### 5.2 Check user's input similarity towards DB
    # if data_dict.loc[0, "TableName1"] is not None:
    #     print("Return Saved Table Detected, getting 3 results")
    #     results_table = pinecone_table_search(input, k=3, filter={"table_description": {"$ne":f"{data_dict.loc[0, 'TableName1']}"}})
    #     intent_db_sim_score = []
    #     table_name = []
    #     table_desc = []
    #     for i in results_table:
    #         temp_intent_db_sim_score,temp_table_name,temp_table_desc = extract_results([i])
    #         print("Table found (name): ", temp_table_name)
    #         print("Score:", temp_intent_db_sim_score)
    #         print("Table found (description): ", temp_table_desc)
    #         intent_db_sim_score.append(temp_intent_db_sim_score)
    #         table_name.append(temp_table_name)
    #         table_desc.append(temp_table_desc)

    # else:
    #     print("Return saved table not detected, getting one result")
    #     results_table = pinecone_table_search(input)
    #     intent_db_sim_score, table_name, table_desc = extract_results(results_table)
    #     print(intent_db_sim_score, table_name, table_desc[:200])

    ### 5.3 Main Logic given input
    # if input == '"--reset--"':
    #     reset_conversation(session_id=session_id)
    #     return "Conversation has been reset."

    # elif data_dict.loc[0,"ChatStage"] == 'inquiry':
    #     test_input_embedding = embeddings.embed_query(input)
    #     list_of_inquiry_embeddings=[inquiry_embedding_1,inquiry_embedding_2,inquiry_embedding_3,inquiry_embedding_4,inquiry_embedding_5,inquiry_embedding_6,inquiry_embedding_7,inquiry_embedding_8,inquiry_embedding_9]
    #     list_of_non_inquiry_embeddings=[non_inquiry_embedding_1,non_inquiry_embedding_2,non_inquiry_embedding_3,non_inquiry_embedding_4,non_inquiry_embedding_5,non_inquiry_embedding_6,non_inquiry_embedding_7,non_inquiry_embedding_8,non_inquiry_embedding_9,non_inquiry_embedding_10]
    #     similarity_inquiry = max([cosine_similarity(test_input_embedding, inquiry_embedding) for inquiry_embedding in list_of_inquiry_embeddings])
    #     similarity_non_inquiry = max([cosine_similarity(test_input_embedding, non_inquiry_embedding) for non_inquiry_embedding in list_of_non_inquiry_embeddings])
    #     if similarity_inquiry > similarity_non_inquiry: ## If we define that it is an inquiry
    #         column_description = pinecone_column_search(userInputQuestion=input ,filter={"sql_path": {"$eq":f"{data_dict.loc[0, 'table_choice']}"}})
    #         messagesForInquiry.append({"role":"user","content":f"Relevant Column Description is Found: {column_description}. Be sure to follow the format provided"})
    #         chat = openai.ChatCompletion.create(
    #             engine="gpt-35-turbo",
    #             messages=messagesForInquiry
    #         )
    #         temp_saved_table_name=data_dict.loc[0,"table_choice"]
    #         returned_saved_input=data_dict.loc[0,"previous_intent"]
    #         return [temp_saved_table_name,
    #             returned_saved_input,
    #             chat.choices[0].message.content]
    #     else: ## If we determine that it is NOT an inquiry
    #         reply_result_denied = "I'm sorry, it seems like this is not an inquiry. Feel free to inquire about any of the columns in the table. If you would like to reset the conversation, type '--reset--'"
    #         temp_saved_table_name=data_dict.loc[0,"table_choice"]
    #         returned_saved_input=data_dict.loc[0,"previous_intent"]
    #         return [temp_saved_table_name,
    #             returned_saved_input,
    #             reply_result_denied]

    # elif data_dict.loc[0,"TableName1"] is not None and data_dict.loc[0,"ChatStage"] == 'confirming': #### Runs when there is a saved table name, meaning previous chat had a table > 0.78
    #     if data_dict.loc[0,"TableName2"] is None and data_dict.loc[0,"TableName3"] is None: #### If prev chat table > 0.78 AND is the first time (returns 1 result)
    #         # cari user confirmation
    #         similarity_yes = cosine_similarity(yes_embedding, embeddings.embed_query(input))
    #         similarity_no = cosine_similarity(no_embedding, embeddings.embed_query(input))
                    
    #         if similarity_yes > similarity_no: #### If more Yes than No
    #             print("Choice accepted")
    #             messagesForResultAccepted = [message for message in messagesForResultAccepted if message['role'] != 'user']
    #             messagesForResultAccepted.append({"role":"user", "content":f"Table to analyze: {saved_table_desc}"})
    #             chat = openai.ChatCompletion.create(
    #                 engine="gpt-35-turbo",
    #                 messages=messagesForResultAccepted
    #             )
    #             reply_result_accepted=clean_response(chat.choices[0].message.content)

    #             messagesForResult = [message for message in messagesForResult if message['role'] != 'user']
    #             driving_message = [message for message in driving_message if message['role'] == 'system']
                
    #             temp_saved_table_name=data_dict.loc[0,"TableName1"]
    #             returned_saved_input=data_dict.loc[0,"previous_intent"]
                
    #             saved_table_name=""
    #             update_table_into_sql(val1=None, val4='inquiry', val6=session_id, val7=temp_saved_table_name)
    #             saved_table_desc=""
    #             saved_input=""

    #             return [temp_saved_table_name,
    #                     returned_saved_input,
    #                     reply_result_accepted]

    #         else:    #### If more no than yes
    #             print("Message denied")
    #             messagesForResultDenied = [message for message in messagesForResultDenied if message['role'] != 'user']
    #             print(messagesForResultDenied)
    #             chat = openai.ChatCompletion.create(
    #                 engine="gpt-35-turbo",
    #                 messages=messagesForResultDenied
    #             )
    #             reply_result_denied=chat.choices[0].message.content

    #             messagesForResult = [message for message in messagesForResult if message['role'] != 'user']
    #             driving_message = [message for message in driving_message if message['role'] == 'system']
                
    #             saved_table_desc=""
    #             saved_input=""
    #             update_table_into_sql(val1=data_dict.loc[0,"TableName1"], val4='driving', val6=session_id)
    #             driving_message.append({"role":"user", "content": input})
    #             driving_message.append({"role":"assistant", "content": reply_result_denied})
    #             returned_saved_input=saved_input

    #             return reply_result_denied
    #     else: #### If previous chat table > 0.78 and returns more than 1 result (not first round)
    #         print("saved_table_name is a list, choosing 1 of 3")
    #         question = f'''
    #         I found {len(table_name)} tables that might relate to Chris's query.
    #         Below are the table(s) that I found:
    #         {table_desc}

    #         Below is Chris's response towards my finding:
    #         {input}

    #         Keep in mind that Chris might answer according to the table's name or table's number/sequence. Which table does Chris wants?

    #         Sample of Expected Answer 1:
    #         {{"chris_wants":"table 1"}}
    #         Sample of Expected Answer 2:
    #         {{"chris_wants":"table 2"}}
    #         Sample of Expected Answer 3:
    #         {{"chris_wants":"table 3"}}
    #         Sample of Expected Answer 4:
    #         {{"chris_wants":"None"}}

    #         !!!IMPORTANT!!! DO NOT GIVE ANSWER OUTSIDE THE 4 SAMPLE OF EXPECTED ANSWER!!!
    #         Answer (Please give it in a dictionary format like the sample of expected answers above):
    #         '''

    #         llm = AzureOpenAI(
    #             temperature=0.1,
    #             verbose=True,
    #             deployment_name='gpt-35-turbo',
    #             model_name='gpt-35-turbo',
    #             max_tokens=3900-num_tokens_from_string(question,'gpt-3.5-turbo'))

    #         response=llm.predict(question)
    #         index_of_brace = response.find('}')

    #         if index_of_brace != -1:
    #             response = response[:index_of_brace + 1]
    #         else:
    #             response = response
    #         eval_message = eval(response)
    #         print("Table desc", table_desc)
    #         print("EVAL MESSAGE",eval_message)
    #         if eval_message['chris_wants'] == "None": #### If message implies none are fit
    #             messagesForResultDeniedFinal = [message for message in messagesForResultDeniedFinal if message['role'] != 'user']
    #             messagesForResultDeniedFinal.append({"role":"user", "content": input})
    #             chat = openai.ChatCompletion.create(
    #                 engine="gpt-35-turbo",
    #                 messages=messagesForResultDeniedFinal
    #             )   
    #             reply_result_denied_final=chat.choices[0].message.content
    #             print("Picked NONE of the 3 tables")
    #             reset_conversation(session_id=session_id)
    #             saved_table_desc=""
    #             saved_input=""

    #             return reply_result_denied_final
    #         else: #### If message implies one of them fit
    #             match = int(eval_message['chris_wants'][-1])
    #             print("Messages for result accepted",messagesForResultAccepted)
    #             messagesForResultAccepted = [message for message in messagesForResultAccepted if message['role'] != 'user']
    #             messagesForResultAccepted.append({"role":"user", "content":f"Table to analyze: {saved_table_desc}"})
    #             chat = openai.ChatCompletion.create(
    #                 engine="gpt-35-turbo",
    #                 messages=messagesForResultAccepted
    #             )   
    #             reply_result_accepted=clean_response(chat.choices[0].message.content)
    #             print("reply_result_accepted")
    #             print(reply_result_accepted)

    #             messagesForResult = [message for message in messagesForResult if message['role'] != 'user']
    #             driving_message = [message for message in driving_message if message['role'] == 'system']
                
    #             returned_saved_input=data_dict.loc[0,"previous_intent"]
    #             temp_saved_table_name = data_dict.loc[0,f"TableName{match}"]
    #             update_table_into_sql(val1=None, val4='inquiry', val6=session_id, val7=temp_saved_table_name)
    #             saved_table_desc=""
    #             saved_input=""
                
    #             return [temp_saved_table_name,
    #                     returned_saved_input,
    #                     reply_result_accepted]
        
    # elif (isinstance(intent_db_sim_score, float) and intent_db_sim_score > 0.78) or (isinstance(intent_db_sim_score, list) and all(isinstance(item, float) and item > 0.74 for item in intent_db_sim_score)) :
    #     if isinstance(intent_db_sim_score, float): #First round, only looking for 1 answer
    #         messagesForResult.append({"role":"user","content":f"Relevant Table is Found: {table_desc}. Be sure to follow the format provided"})
    #         chat = openai.ChatCompletion.create(
    #             engine="gpt-35-turbo",
    #             messages=messagesForResult
    #         )
    #         reply_table_found=chat.choices[0].message.content
    #         print(messagesForResult)
            
    #         # saved_table_name = table_name
    #         update_table_into_sql(val1=table_name, val4='confirming', val5=input, val6=session_id)
    #         saved_table_desc = table_desc
    #         saved_input = input
            
    #         returned_saved_table_name = saved_table_name
    #         previous_table_desc = table_desc
    #         print("giving user the choice of 1 table")
    #         return reply_table_found
    #     else: #Second round, looking for 3 answers
    #         messagesForMultipleResults.append({"role":"user","content":f"Relevant Tables are Found: {table_desc}. Be sure to follow the format provided"})
    #         chat = openai.ChatCompletion.create(
    #             engine="gpt-35-turbo",
    #             messages=messagesForMultipleResults
    #         )
    #         reply_table_found=chat.choices[0].message.content
    #         ### HARDCODED FOR NOW, MUST REFACTOR WHEN SCALING
    #         try:
    #             val1 = table_name[0]
    #         except:
    #             print("table_name[0] not found")
    #             val1 = None
    #         try:
    #             val2 = table_name[1]
    #         except:
    #             print("table_name[1] not found")
    #             val2 = None
    #         try:
    #             val3 = table_name[2]
    #         except:
    #             print("table_name[3] not found")
    #             val3 = None    

    #         update_table_into_sql(val1=val1, val2=val2, val3=val3, val4='confirming', val5=input, val6=session_id)
    #         print("giving user the choice of 3 table")
    #         saved_table_desc = table_desc
    #         saved_input = input
    #         return reply_table_found

    # else:
    #     # Intent Not Found
    #     driving_message.append({"role":"user", "content": input})
    #     chat = openai.ChatCompletion.create(
    #         engine="gpt-35-turbo",
    #         messages=driving_message
    #     )
    #     reply_driving_message = chat.choices[0].message.content
    #     driving_message.append({"role":"assistant", "content": reply_driving_message})
    #     print("Driving message")
    #     print(driving_message)
    #     return reply_driving_message

    if input == '--reset--':
        driving_message_temp = reset_conversation(session_id=session_id)
        driving_message = driving_message_temp
        return "Conversation has been reset"
        
    elif data_dict.loc[0,"ChatStage"] == 'driving':
        intents = ["inquire one or many tables","not an inquiry", "show list of available tables"]
        intent = intent_classifier(input=input, intents=intents)
        
        if intent == 'inquire one or many tables':
            intent_db_sim_score, table_name, table_desc = get_results_for_table(client_Id=client_Id, data_dict=data_dict, input=input)
            if (isinstance(intent_db_sim_score, float) and intent_db_sim_score > 0.78) or (isinstance(intent_db_sim_score, list) and all(isinstance(item, float) and item > 0.74 for item in intent_db_sim_score)) :
                reply_table_found, saved_table_desc, saved_input = results_table_found(input=input, intent_db_sim_score=intent_db_sim_score, table_name=table_name, table_desc=table_desc, session_id=session_id)

                return f'Client ID: {client_Id}\n\n'+reply_table_found
            else:
                reply_driving_message = get_driving_message(input=input, driving_message=driving_message)
                return f'Client ID: {client_Id}\n\n'+reply_driving_message
        elif intent == 'not an inquiry':
            reply_driving_message = get_driving_message(input=input, driving_message=driving_message)
            return f'Client ID: {client_Id}\n\n'+reply_driving_message
        elif intent == 'show list of available tables':
            return(f'Client ID: {client_Id}\n\n'+pinecone_table_list_search(client_Id=client_Id))
        else:
            return(f'Client ID: {client_Id}\n\n'+"Unidentified intent! Check intent classifier! Staying on Driving stage!")

    elif data_dict.loc[0,"ChatStage"] == 'confirming_single':
        intents = ["yes", "no", "no and then user suggests a new table", "not answering the question"]
        intent = intent_classifier(input=input, intents=intents)
        print("Intent for confirming_single phase: ", intent)
        if intent == 'yes':
            return_list = accept_single_table_choice(data_dict=data_dict, session_id=session_id)
            messagesForResult = [message for message in messagesForResult if message['role'] != 'user']
            driving_message = [message for message in driving_message if message['role'] == 'system']
            if not check_for_dimension(client_Id=client_Id, data_dict=data_dict, input=input, table_choice=data_dict.loc[0,'TableName1']):  
                return return_list
            table_choice = data_dict.loc[0,'TableName1']
            table_choice_desc = data_dict.loc[0, 'TableDesc1']
            driving_input = data_dict.loc[0, 'previous_intent']
            update_table_into_sql(val4='refining_pt1' ,val5=driving_input ,val6=session_id, val7=table_choice, val8=table_choice_desc)
            return(f'Client ID: {client_Id}\n\n'+table_dimensions_found(client_Id=client_Id, input=table_choice))

        elif intent == 'no':
            return_list = f'Client ID: {client_Id}\n\n'+reject_single_table_choice(data_dict=data_dict, session_id=session_id)
            return return_list
        elif intent == 'no and then user suggests a new table':
            return(f'Client ID: {client_Id}\n\n'+"No, but user suggests a new intent. Not impleted yet, will stay in confirming phase. Please type yes or no")
        elif intent == 'not answering the question':
            return(f'Client ID: {client_Id}\n\n'+"Not answering the question, but user suggests a new intent. Not impleted yet, will stay in confirming phase. Please type yes or no")
        else:
            return(f'Client ID: {client_Id}\n\n'+"Unidentified intent! Check intent classifier! Staying on Confirming stage!")

    elif data_dict.loc[0,"ChatStage"] == 'confirming_multiple':
        return_list, saved_table_desc = multiple_table_choice(input=input, data_dict=data_dict, session_id=session_id, client_Id=client_Id)
        if isinstance(return_list, list):
            #Chosen either 1,2,3
            if not check_for_dimension(client_Id=client_Id, data_dict=data_dict, input=input, table_choice=data_dict.loc[0,'TableName1']):  
                return return_list
            else:
                # breakpoint()
                table_choice = return_list[1]
                table_choice_desc = saved_table_desc
                driving_input = data_dict.loc[0, 'previous_intent']
                update_table_into_sql(val4='refining_pt1' ,val5=driving_input ,val6=session_id, val7=table_choice, val8=table_choice_desc)
                return(table_dimensions_found(client_Id=client_Id, input=table_choice))
        elif isinstance(return_list, str):
            #Chosen None
            return return_list
        

        # return return_list

    elif data_dict.loc[0,"ChatStage"] == 'refining_pt1':
        # By default we assume there is a refine intention, run pinecone sim search. If none, then mention
        intents = ["refine search", "not an inquiry"]
        intent = intent_classifier(input=input, intents=intents)

        if intent == 'refine search':
            # result_list = accept_single_table_choice(data_dict=data_dict, session_id=session_id)
            filter_dimension, result = generate_dimension_filter_and_items(client_Id=client_Id, data_dict=data_dict, input=input, messageForRefinementPossible=messageForRefinementPossible)
            table_choice=data_dict.loc[0,"table_choice"]
            table_choice_desc=data_dict.loc[0,"table_choice_desc"]
            driving_input = data_dict.loc[0,"previous_intent"]
            update_table_into_sql(val4='refining_pt2' ,val5=driving_input ,val6=session_id, val7=table_choice, val8=table_choice_desc, val12=filter_dimension)
            return (f'Client ID: {client_Id}\n\n'+result)
        elif intent == 'not an inquiry':
            temp_saved_table_name = data_dict.loc[0,"table_choice"]
            temp_saved_table_desc = data_dict.loc[0,"table_choice_desc"]
            original_input = data_dict.loc[0,"previous_intent"]
            reply_refine_not_required = "It seems that no refining is requested. I will proceed with the analytics for the complete table."
            empty_string = "{}"
            update_table_into_sql(val1=None, val4='driving', val6=session_id, val7=temp_saved_table_name, val8=temp_saved_table_desc)
            return [{'reloadSixups': True},
                temp_saved_table_name,
                original_input,
                empty_string,
                client_Id,
                user_Id,
                f'Client ID: {client_Id}\n\n'+reply_refine_not_required]
        
    elif data_dict.loc[0,"ChatStage"] == 'refining_pt2':
        # This is where we put in the filter values that we want to refine by
        #1. Check if the value exists using Guidance
        #2. If yes, output {filter_dimension:{'product_id': ['Mountain']}}
        #3. If no, stay on refining pt_2, but return a value not identified
        refine_dimension = data_dict.loc[0,'refine_dimension']
        filter_items = generate_dimension_filter_and_items(client_Id=client_Id, data_dict=data_dict, input=input, refine_dimension=refine_dimension)
        answer = guidance_filter_item_identifier(input, filter_items)
        temp_saved_table_name = data_dict.loc[0,"table_choice"]
        temp_saved_table_desc = data_dict.loc[0,"table_choice_desc"]
        original_input = data_dict.loc[0,"previous_intent"]
        empty_string = "{}"
        #Special check for when user decides not to have filters anymore
        if answer == "--None--":
            update_table_into_sql(val1=None, val4='driving', val6=session_id, val7=temp_saved_table_name, val8=temp_saved_table_desc)
            reply_refine_not_required = "It seems that refining is no longer required. I will proceed with the analytics for the complete table."
            return [{'reloadSixups': True},
                temp_saved_table_name,
                original_input,
                empty_string,
                client_Id,
                user_Id,
                f'Client ID: {client_Id}\n\n'+reply_refine_not_required]
        filter_dimension = {refine_dimension: process_input(answer)}
        json_string = json.dumps({'filter_dimension': filter_dimension})
        messagesForResultAccepted = [message for message in messagesForResultAccepted if message['role'] != 'user']
        messagesForResultAccepted.append({"role":"user", "content":f"Table to analyze: {temp_saved_table_desc}"})
        chat = openai.ChatCompletion.create(
            engine="gpt-35-turbo",
            messages=messagesForResultAccepted
        )
        reply_result_accepted=clean_response(chat.choices[0].message.content)
        update_table_into_sql(val1=None, val4='driving', val6=session_id, val7=temp_saved_table_name, val8=temp_saved_table_desc)
        return [{'reloadSixups': True},
                temp_saved_table_name,
                original_input,
                json_string,
                client_Id,
                user_Id,
                f'Client ID: {client_Id}\n\n'+reply_result_accepted]

    elif data_dict.loc[0,'ChatStage'] == 'inquiry':
        table_sim_score, table_name, table_desc= get_results_for_table(client_Id=client_Id, data_dict=data_dict, input=input)
        column_sim_score = pinecone_column_search(client_Id=client_Id, userInputQuestion=input, filter={"table_description": {"$ne":f"{data_dict.loc[0, 'TableName1']}"}})['matches'][0]['score']
        print("Column sim score: ",column_sim_score)
        print("Table sim score: ",table_sim_score)
        if column_sim_score > 0.85:
            #Assume its column similarity search
            return_list = inquire_column(data_dict=data_dict, input=input, client_Id=client_Id)
            return(return_list)
        elif table_sim_score > 0.78:
            #Assume its table similarity search
            reply_table_found, saved_table_desc, saved_input = results_table_found(input=input, intent_db_sim_score=table_sim_score, table_name=table_name, table_desc=table_desc, session_id=session_id)
            return reply_table_found
        else:
            #Not an inquiry or asking for mathematical calculations
            intents = ["asking for mathematical calculations", "not an inquiry"]
            intent_layer2 = intent_classifier(input=input, intents=intents)
            if intent_layer2 == "asking for mathematical calculations":
                return[data_dict.loc[0,"table_choice"],
                       input,
                        "Asking for math calculations"]
            elif intent_layer2 == "not an inquiry":
                reply_driving_message = get_inquiry_driving_message(input=input, data_dict=data_dict)
                return [data_dict.loc[0,"table_choice"],
                        input,
                        reply_driving_message]
            else:
                return("Error in layer 2 of intent classification. Staying in inquiry phase.")
##### Section 6: FastAPI code #####    
app = FastAPI()
    
allowed_origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,  # If you want to allow cookies to be sent
#    allow_methods=["*"],  # Specify the HTTP methods you want to allow
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],  # You can customize the allowed headers or use "*" to allow any header
)
@app.get('/')
async def root():
    return{'example':'This is test','data':0}

@app.get("/chat")
async def chat(input_text: str, session_id: str, client_Id: str, user_Id: str):
    ### IMPORTANT! ONLY USE THIS FOR DEVELOPMENT
    try:
        reply = chatbot(input_text, session_id, client_Id, user_Id) 
        print("REPLY: ", reply)
        return reply   
    except Exception as e:
        return traceback.format_exc()
    ### IMPORTANT! COMMENT OUT THE OTHER ONE AND USE THIS FOR DEMO
    try:
        reply = chatbot(input_text, session_id, client_Id, user_Id) 
        print("REPLY: ", reply)
        return reply   
    except Exception as e:
        print(traceback.format_exc())

@app.get("/checkDb")
async def checkDb(session_id: str):
    host = "20.212.32.214"
    user = "UserDataDb"
    password = "sa_54321"
    database = "datamart"
    connection = mysql.connector.connect(host=host,
                                        database=database,
                                        user=user,
                                        password=password)
    query_df = f"SELECT * FROM datamart.BrahmaputraTableDev WHERE session_id='{session_id}';"
    cursor = connection.cursor()
    cursor.execute(query_df)
    # get all records
    records = cursor.fetchall()
    tableNames = ('Table1', 'Table2', 'Table3', 'ChatStage', 'previous_intent', 'session_id')
    result = zip(tableNames, records[0])
    result_list = []
    connection.close()
    for i in result:
        result_list.append(i)
    return result_list

if __name__ == "__main__":
    import uvicorn
    #uvicorn.run(app, host="192.168.100.80", port=8080)
    #uvicorn.run(app, host="192.168.30.228", port=8080)
    uvicorn.run(app, host="0.0.0.0", port=8000)