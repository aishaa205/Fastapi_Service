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

DATABASE_URL = f"postgresql+asyncpg://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"

engine = create_async_engine(DATABASE_URL, echo=True, 
    connect_args={"statement_cache_size": 0}
)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

metadata = MetaData()

# Reflect an existing table
tables = ['answers_answer', 'applications_application', 'jobs_job', 'questions_question', 'user_user']
async def get_user_table(table_name):
    if table_name not in tables:
        raise ValueError(f"Invalid table name: {table_name}")
    async with engine.begin() as conn:
        await conn.run_sync(metadata.reflect)
    return Table(table_name, metadata, autoload_with=engine)


# @app.patch("/users/{user_id}")
# async def patch_user(user_id: int, new_name: str, db: AsyncSession = Depends(lambda: async_session())):
#     user_table = await get_user_table("user_user")
#     stmt = (
#         update(user_table)
#         .where(user_table.c.id == user_id)
#         .values(name=new_name)
#         .execution_options(synchronize_session="fetch")
#     )
#     await db.execute(stmt)
#     await db.commit()
#     return {"status": "updated"}