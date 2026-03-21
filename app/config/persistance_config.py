from datetime import datetime
from pymongo import MongoClient, ASCENDING
import os
from dotenv import load_dotenv

load_dotenv()

mongo_client = MongoClient(os.getenv("MONGO_DB_URI"))
db = mongo_client["rl_db"]
experiences_collection = db["experiences"]

experiences_collection.create_index(
    [("env_name", ASCENDING), ("timestamp", ASCENDING)]
)