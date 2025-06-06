from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

import pyodbc
import datetime
from typing import List, Dict, Union, Any
import re
import pandas as pd

from pydantic import BaseModel


from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware, 
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "/app/llama3.2_1b"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def get_db_connection():
    conn_str = (
        r'DRIVER={FreeTDS};'
        r'SERVER=10.73.131.12;'
        r'PORT=1433;'
        r'DATABASE=JidoukaProject;'
        r'UID=intern;'
        r'PWD=intern1234qwer!;'
        r'TrustServerCertificate=Yes;'
    )
    return pyodbc.connect(conn_str)

mydb = get_db_connection()


class ExecuteQuery:
    def __init__(self, db):
        self.db = db
    def is_valid_sql_query(self, query: str) -> bool:

        if query.upper().startswith('SELECT'):
            return True 
        return False 
    def convert_sql_query(self,query: str) -> str:
        """
        Convert SQL query to be compatible with Microsoft SQL Server syntax
        
        Args:
            query (str): Original SQL query
        
        Returns:
            str: Converted SQL query compatible with Microsoft SQL Server
        """
        # Replace LIMIT with TOP
        if 'LIMIT' in query.upper():
            # Extract the LIMIT value
            limit_match = re.search(r'LIMIT\s+(\d+)', query, re.IGNORECASE)
            if limit_match:
                limit_value = limit_match.group(1)
                # Replace LIMIT with TOP
                query = re.sub(r'LIMIT\s+\d+', '', query, flags=re.IGNORECASE)
                
                # Check if SELECT is the first word
                if query.upper().startswith('SELECT'):
                    query = query.replace('SELECT', f'SELECT TOP {limit_value}', 1)
                else:
                    # If SELECT is not at the start, we'll add TOP manually
                    query = f'SELECT TOP {limit_value} {query[6:]}'
        
        # Modify column names
        query = query.replace('ImprovementName', 'ImprovementContent')
        query = query.replace('TimeSaving', 'TotalTimeSaved')
        
        return query

    def sanitize_query(self, query: str) -> str:
        query = re.sub(r'/\*.*?\*/', '', query)
        query = re.sub(r'--.*$','', query)
        query = query.strip()
        #query = query.sub('LIMIT1', '', query)
        query = self.convert_sql_query(query)
        query = re.sub('ImprovementName', 'ImprovementContent', query)
        query = re.sub('TotalTimeSaved', 'TimeSaving', query)


        return query 
    
    def execute_query(self, query: str) -> Union[List[Dict[str, Any]], str]:
        if self.is_valid_sql_query(query):
            sanitized_query = self.sanitize_query(query)
            df = pd.read_sql_query(sanitized_query, self.db)

            if df.columns == ['']:
                return f"Thông tin bạn cần là: {str(df.iloc[0,0])}"
            if len(df.columns) == 1:
                return f"{str(df.columns[0])}: {str(df.iloc[0,0])}"
            return df
        else: 
            return query

class JidoukaModel:
    def __init__(self, max_history: int=0):
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.max_history = max_history
        self.conversation_history: List[Dict[str, str]] = []
    
    def _build_prompt(self) -> str:
        # Build context from history
        if self.max_history == 0:
            system_prompt = """You are an SQL query assistant. Based on schema, generate an SQL query to retrieve the relevant information for the user. If the user’s question is unrelated to the table, respond naturally in user's language.
            Schema:
            +Table Author, columns=[AuthorId: int, AuthorName: nvarchar(255), DepartmentId int, GroupDCId int]
            +Table Department, columns=[DepartmentId: int, DepartmentName: nvarchar(255)]
            +Table GroupDC, columns=[GroupDCId: int, DepartmentId: int, GroupDCName nvarchar(255)]
            +Table Job, columns=[JobId: int, JobName: nvarchar(255)]
            +Table Tool, columns=[ToolId: int, ToolName: nvarchar(255), ToolDescription: text]
            +Table Jidouka, columns=[JidoukaId: bigint, ProductApply: nvarchar(255), ImprovementName: nvarchar(255), SoftwareUsing: nvarchar(255), Description: nvarchar(255), Video: text, DetailDocument: text, TotalJobApplied: int, TotalTimeSaved: int, DateCreate: datetime, JobId: int, AuthorId: int, DepartmentId: int, GroupDCId: int]
            +Table JidoukaTool, columns=[JidoukaId: bigint, ToolId: int]
            +Primary_keys=[Author.AuthorId, Department.DepartmentId, GroupDC.GroupDCId, Job.JobId, Tool.ToolId, Jidouka.JidoukaId]
            +Foreign_keys=[GroupDC.DepartmentId=Department.DepartmentId, Jidouka.JobId=Job.JobId, Jidouka.AuthorId=Author.AuthorId, Jidouka.DepartmentId=Department.DepartmentId, Jidouka.GroupDCId=GroupDC.GroupDCId, JidoukaTool.JidoukaId=Jidouka.JidoukaId, JidoukaTool.ToolId=Tool.ToolId, Author.DepartmentId=Department.DepartmentId, Author.GroupDCId=GroupDC.GroupDCId]
            """
            return system_prompt 
        else:
            history_text = ""
            for exchange in self.conversation_history[-self.max_history:]:
                history_text += f"Human: {exchange['human']}\nAssistant: {exchange['assistant']}\n\n"
            # Create the full prompt with context
            prompt = f"""You are an SQL query assistant. Based on schema and history context below, generate an SQL query to retrieve the relevant information for the user. If the user’s question is unrelated to the table, respond naturally in user's language.
            Schema:
            +Table Author, columns=[AuthorId: int, AuthorName: nvarchar(255), DepartmentId int, GroupDCId int]
            +Table Department, columns=[DepartmentId: int, DepartmentName: nvarchar(255)]
            +Table GroupDC, columns=[GroupDCId: int, DepartmentId: int, GroupDCName nvarchar(255)]
            +Table Job, columns=[JobId: int, JobName: nvarchar(255)]
            +Table Tool, columns=[ToolId: int, ToolName: nvarchar(255), ToolDescription: text]
            +Table Jidouka, columns=[JidoukaId: bigint, ProductApply: nvarchar(255), ImprovementName: nvarchar(255), SoftwareUsing: nvarchar(255), Description: nvarchar(255), Video: text, DetailDocument: text, TotalJobApplied: int, TotalTimeSaved: int, DateCreate: datetime, JobId: int, AuthorId: int, DepartmentId: int, GroupDCId: int]
            +Table JidoukaTool, columns=[JidoukaId: bigint, ToolId: int]
            +Primary_keys=[Author.AuthorId, Department.DepartmentId, GroupDC.GroupDCId, Job.JobId, Tool.ToolId, Jidouka.JidoukaId]
            +Foreign_keys=[GroupDC.DepartmentId=Department.DepartmentId, Jidouka.JobId=Job.JobId, Jidouka.AuthorId=Author.AuthorId, Jidouka.DepartmentId=Department.DepartmentId, Jidouka.GroupDCId=GroupDC.GroupDCId, JidoukaTool.JidoukaId=Jidouka.JidoukaId, JidoukaTool.ToolId=Tool.ToolId, Author.DepartmentId=Department.DepartmentId, Author.GroupDCId=GroupDC.GroupDCId]
            History context:
            <START_OF_HISTORY_CONTEXT>
            {history_text}
            <END_OF_HISTORY_CONTEXT>
            """
            return prompt    
    def chat(self, user_input: str) -> str:
        # Generate the contextualized prompt
        prompt = self._build_prompt()
        eot = "<|eot_id|>"
        eot_id = self.tokenizer.convert_tokens_to_ids(eot)
        self.tokenizer.pad_token = eot
        self.tokenizer.pad_token_id = eot_id

        messages =[
            {'role':'system',
             'content':prompt}
            ,
            {'role':'user',
             'content':user_input}
        ]
        chat = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(chat, padding=True, truncation=True, return_tensors='pt').to(DEVICE)
        outputs = self.model.generate(inputs['input_ids'],
                                      attention_mask=inputs['attention_mask'],
                                      temperature = 0.1, 
                                      do_sample = True,
                                      top_p = 0.1
                                      ,max_new_tokens=512).to(DEVICE)
        bot_response = self.tokenizer.decode(outputs[0])
        bot_response = bot_response.split('<|start_header_id|>assistant<|end_header_id|>')
        bot_response = bot_response[1].strip()[:-10]
        # Update conversation history
        self.conversation_history.append({
            'human': user_input,
            'assistant': bot_response
        })
        
        return bot_response
    
chatbot = JidoukaModel()
query_agent = ExecuteQuery(mydb)
chat_history = []

class UserInput(BaseModel):
    message: str

@app.post("/generate")
async def generate(input: UserInput):
    # Parse JSON từ request body
    user_message = input.message
    
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")

    sql_query = chatbot.chat(user_message)

    query_result = query_agent.execute_query(sql_query)
    
    # Prepare response
    if isinstance(query_result, list):
        # If query returns results, convert to readable string
        response = [str(record) for record in query_result.to_dict('records')]
    else:
        # If query fails or returns error
        response = query_result

    chat_history.append({'timestamp': timestamp, 'bot': response})


    
    return {'timestamp': timestamp, 'response': response}

# app.mount("/", StaticFiles(directory="static", html=True), name="static")

# @app.get("/")
# def index() -> FileResponse:
#     return FileResponse(path="/app/static/index.html", media_type="text/html")




