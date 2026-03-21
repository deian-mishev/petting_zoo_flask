from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Optional
from collections import deque

import eventlet
from app.config.env_config import MEMORY_SIZE
import tensorflow as tf

@dataclass
class Experience:
    state: any
    action: int = None
    reward: float = None
    next_state: any = None
    done: bool = None

class PlayerType(str, Enum):
    COMPUTER = "ai_regular",
    ATARI_PRO = "atari_ai_pro",
    HUMAN = "human"

@dataclass
class PlayerState:
    type: PlayerType = None
    model_path: str = None
    weights_path: str = None
    q_network: tf.keras.Model = None
    target_q_network: tf.keras.Model = None
    optimizer: tf.keras.optimizers.Optimizer = None
    lock: eventlet.semaphore.Semaphore = None
    current_experience: Optional[Experience] = None
    total_reward: int = field(default=0)
    memory_buffer: Deque[Experience] = field(
        default_factory=lambda: deque(maxlen=MEMORY_SIZE))
