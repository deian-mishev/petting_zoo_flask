from pymongo import ASCENDING
from app.config.persistance_config import experiences_collection
from app.config.player_state import Experience
from datetime import datetime, timezone
from app import app


class ExperienceService:
    def __init__(self):
        self.collection = experiences_collection

    def insert_experience_batch(self, env_name: str, experiences: list[Experience], sid):
        if not experiences:
            return
        try:
            docs = [{
                "env_name": env_name,
                "state": exp.state.tolist(),
                "action": int(exp.action),
                "reward": float(exp.reward),
                "next_state": exp.next_state.tolist(),
                "done": bool(exp.done),
                "timestamp": datetime.now(timezone.utc)
            } for exp in experiences]
            self.collection.insert_many(docs)
            self.enforce_limit(env_name)
            app.logger.info(
                f"{sid}: Stored {len(experiences)} experiences for {env_name}")
        except Exception as e:
            app.logger.exception(f"{sid}: Failed to store experiences: {e}")

    def insert_experience(self, env_name: str, experience: Experience):
        if not experience:
            return
        doc = {
            "env_name": env_name,
            "state": experience.state.tolist(),
            "action": int(experience.action),
            "reward": float(experience.reward),
            "next_state": experience.next_state.tolist(),
            "done": bool(experience.done),
            "timestamp": datetime.now(timezone.utc)
        }
        self.collection.insert_one(doc)

    def enforce_limit(self, env_name: str, max_entries: int = 10_000):
        pipeline = [
            {"$match": {"env_name": env_name}},
            {"$sort": {"timestamp": ASCENDING}},
            {"$skip": max_entries},
            {"$project": {"_id": 1}}
        ]
        docs_to_delete = list(self.collection.aggregate(pipeline))
        if docs_to_delete:
            ids_to_delete = [doc["_id"] for doc in docs_to_delete]
            self.collection.delete_many({"_id": {"$in": ids_to_delete}})

    def sample_experiences(self, env_name: str, batch_size: int) -> list[Experience]:
        cursor = self.collection.aggregate([
            {"$match": {"env_name": env_name}},
            {"$sample": {"size": batch_size}}
        ])
        experiences = []
        for doc in cursor:
            experiences.append(Experience(
                state=doc["state"],
                action=doc["action"],
                reward=doc["reward"],
                next_state=doc["next_state"],
                done=doc["done"]
            ))
        return experiences

experience_service = ExperienceService()
