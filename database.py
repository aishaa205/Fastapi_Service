from pymongo import MongoClient
import os
from dotenv import load_dotenv
load_dotenv()
############Mongo#############
MONGO_URI = "mongodb+srv://omar:1632025@cluster0.yk9bb.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DB_NAME = "job_db"

client = MongoClient(MONGO_URI)

db = client[DB_NAME]
jobs_collection = db["jobs"]
users_collection=db["user_cv_db"]
rag_collection=db["Rag"]
rag_names_collection = db["rag_names"]
test_collection=db["cv_test"]


############Postgres#############
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import MetaData, Table, update
import asyncio
from sqlalchemy.exc import DBAPIError
import logging

DATABASE_URL = f"postgresql+asyncpg://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"

engine = create_async_engine(DATABASE_URL, echo=False, 
    connect_args={"statement_cache_size": 0},
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

metadata = MetaData()

# Reflect an existing table
tables = ['answers_answer', 'applications_application', 'jobs_job', 'questions_question', 'user_user']
logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
async def get_user_table(table_name, retries=3):
    if table_name not in tables:
        raise ValueError(f"Invalid table name: {table_name}")
    for attempt in range(retries):
        try:
            async with engine.begin() as conn:
                metadata = MetaData()
                await conn.run_sync(metadata.reflect)
                return metadata.tables[table_name]
        except DBAPIError as e:
            if attempt < retries - 1:
                await asyncio.sleep(1)
                continue
            raise e