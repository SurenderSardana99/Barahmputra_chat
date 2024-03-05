

import os
import re
import pinecone
import numpy as np
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

#Initialize sql connection
host = "20.212.32.214"
user = "UserDataDb"
password = "sa_54321"
database = "datalake"
connection = mysql.connector.connect(host=host,
                                    database=database,
                                    user=user,
                                    password=password)
query_df = f"SELECT * FROM datalake.BrahmaputraTable;"
# query_df = f"""
# ALTER TABLE datalake.BrahmaputraTable
# ADD COLUMN previous_intent VARCHAR(255);
# """
cursor = connection.cursor()
cursor.execute(query_df)
# get all records
records = cursor.fetchall()
print(records)
