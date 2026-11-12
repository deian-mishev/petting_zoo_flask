import logging
from pymongo import MongoClient, ASCENDING
import os
from dotenv import load_dotenv

load_dotenv()

mongo_enabled_value = os.getenv("MONGO_ENABLED", "true").strip().lower()
if mongo_enabled_value not in {"true", "false"}:
    raise ValueError("MONGO_ENABLED must be 'true' or 'false'")

MONGO_ENABLED = mongo_enabled_value == "true"
mongo_client = None
db = None
experiences_collection = None

if MONGO_ENABLED:
    mongo_uri = os.getenv("MONGO_DB_URI")
    if not mongo_uri or not mongo_uri.strip():
        raise ValueError("MONGO_DB_URI is required when MONGO_ENABLED=true")
    mongo_client = MongoClient(mongo_uri)
    db = mongo_client["rl_db"]
    experiences_collection = db["experiences"]
    experiences_collection.create_index(
        [("env_name", ASCENDING), ("timestamp", ASCENDING)]
    )
else:
    logging.getLogger("app").info("MongoDB experience persistence is disabled.")


def get_experiences_collection():
    if experiences_collection is None:
        raise RuntimeError(
            "MongoDB experience persistence is disabled; set MONGO_ENABLED=true "
            "and configure MONGO_DB_URI to enable it."
        )
    return experiences_collection