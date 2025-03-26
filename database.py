from pymongo import MongoClient
import os

MONGO_URI = "mongodb+srv://omar:1632025@cluster0.yk9bb.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DB_NAME = "job_db"


client = MongoClient(MONGO_URI)

print(client.list_database_names()) 

db = client[DB_NAME]
jobs_collection = db["jobs"]
users_collection=db["user_cv_db"]