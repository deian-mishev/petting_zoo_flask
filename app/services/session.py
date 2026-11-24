from app import app, client_sessions, client_sessions_lock
from app.config.session_state import SessionState
from app.config.ml_env_config import ENVIRONMENTS, EnvironmentConfig
from app.config.env_config import ATTARI_PRO_LOCK, ATTARI_PRO_MODEL, ATTARI_PRO_WEIGHTS_PATH
from app.config.player_state import PlayerState, PlayerType


def cleanup_session(sid, from_field: bool = True):
    with client_sessions_lock:
        session = client_sessions.get(sid)
        if session and getattr(session, '_cleanup_started', False):
            return
        if session:
            session._cleanup_started = True

    if not session:
        app.logger.info(f"{sid}: No session found to clean up.")
        return

    try:
        if from_field and session.runner:
            session.runner.stop()
        try:
            session.env.close()
        except Exception as e:
            app.logger.error(f"{sid}: Error closing env for session: {e}")
        if from_field and session.runner:
            session.runner.update_after_match()
    finally:
        with client_sessions_lock:
            if client_sessions.get(sid) is session:
                del client_sessions[sid]
        app.logger.info(f"{sid}: Session cleaned up.")


def reserve_session(sid, session):
    requested = {agent.type for agent in session.agents.values()}
    with client_sessions_lock:
        if sid in client_sessions:
            return False
        for existing in client_sessions.values():
            active = {agent.type for agent in existing.agents.values()}
            shared = {PlayerType.HUMAN, PlayerType.ATARI_PRO}
            if requested & shared and active & shared:
                return False
            if (PlayerType.COMPUTER in requested and PlayerType.COMPUTER in active
                    and session.env_config.name == existing.env_config.name):
                return False
        client_sessions[sid] = session
        return True

def get_available_environments_and_nemesis():
    with client_sessions_lock:
        used_env_names = set()
        available_players = [pt.value for pt in PlayerType]
        atari_pro_removed = False
        for session in client_sessions.values():
            used_env_names.add(session.env_config.name)
            if not atari_pro_removed:
                for agent in session.agents.values():
                    if agent.type in (PlayerType.HUMAN, PlayerType.ATARI_PRO):
                        available_players.remove(PlayerType.ATARI_PRO.value)
                        available_players.remove(PlayerType.HUMAN.value)
                        atari_pro_removed = True

        available_envs = [
            {"name": name, "agents": cfg.agents}
            for name, cfg in ENVIRONMENTS.items()
            if cfg.name not in used_env_names
        ]
    return available_envs, available_players

def populate_session_agents(session_state: SessionState, env_config: EnvironmentConfig, players: dict[str, str]):
    for key, value in players.items():
        player = PlayerState(type=PlayerType(value))
        if player.type in [PlayerType.HUMAN, PlayerType.ATARI_PRO]:
            player.lock = ATTARI_PRO_LOCK
            player.model_path = ATTARI_PRO_MODEL
            player.weights_path = ATTARI_PRO_WEIGHTS_PATH
        else:
            player.lock = env_config.lock
            player.model_path = env_config.model_path
            player.weights_path = env_config.weights_path
        session_state.agents[key] = player
