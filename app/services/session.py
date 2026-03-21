from app import app, client_sessions, client_sessions_lock
from app.services.ml_service import ml_service
from app.services.experience_store import experience_service
from app.config.session_state import SessionState
from app.config.ml_env_config import ENVIRONMENTS, EnvironmentConfig
from app.config.env_config import ATTARI_PRO_LOCK, ATTARI_PRO_MODEL, ATTARI_PRO_WEIGHTS_PATH
from app.config.player_state import PlayerState, PlayerType


def cleanup_session(sid, from_field: bool = True):
    with client_sessions_lock:
        session = client_sessions.pop(sid, None)

    if not session:
        app.logger.info(f"{sid}: No session found to clean up.")
        return

    if from_field and session.runner:
        session.runner.stop()

    try:
        session.env.close()
    except Exception as e:
        app.logger.error(f"{sid}: Error closing env for session: {e}")

    app.logger.info(f"{sid}: Session cleaned up.")
    # socketio.start_background_task(
    #     experience_service.insert_experience_batch,
    #     env_name=session.env_config.name,
    #     experiences=list(best_score_agent.memory_buffer),
    #     sid=sid
    # )
    if from_field:
        try:
            ml_service.train_model(sid, session)
        except Exception as e:
            app.logger.error(f"{sid}: Error updating model for session: {e}")

def get_available_environments_and_nemesis():
    with client_sessions_lock:
        used_env_names = set()
        available_players = [pt.value for pt in PlayerType]
        atari_pro_removed = False
        for session in client_sessions.values():
            used_env_names.add(session.env_config.name)
            if not atari_pro_removed:
                for agent in session.agents.values():
                    if agent.type == PlayerType.ATARI_PRO:
                        available_players.remove(PlayerType.ATARI_PRO.value)
                        atari_pro_removed = True

            if atari_pro_removed:
                break

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