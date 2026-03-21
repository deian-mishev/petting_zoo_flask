from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional
import eventlet

from app.config.ml_env_config import EnvironmentConfig
from app.config.player_state import PlayerState

if TYPE_CHECKING:
    from app.services.session_runner import SessionRunner

@dataclass
class SessionState:
    env: object
    env_config: EnvironmentConfig
    runner: Optional["SessionRunner"] = None
    agent_iter: Optional[object] = None
    current_agent: Optional[PlayerState] = None
    next_human_action: int = 0
    agents: Dict[str, PlayerState] = field(default_factory=dict)
    lock: eventlet.semaphore.Semaphore = field(default_factory=eventlet.semaphore.Semaphore)