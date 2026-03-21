from flask import jsonify, request

from app import app, client_sessions, client_sessions_lock
from app.services.ml_service import ml_service
from app.services.session_runner import SessionRunner
from app.config.session_state import SessionState
from app.config.ml_env_config import EnvironmentConfig, ENVIRONMENTS
from app.config.env_config import *
from app.config.oauth2_config import token_required_api
from app.validation import validate_env, validate_players
from app.config.player_state import PlayerType
from app.services.session import populate_session_agents


@app.route('/train', methods=['POST'])
@token_required_api(roles=['Admin'])
def train():
    sid = request.user['sub']
    with client_sessions_lock:
        if sid in client_sessions:
            return jsonify({"error": "Training session in progress"}), 400

    user = request.user['email']
    data = request.get_json()
    env_name = data.get("env")
    players = data.get("players")
    episodes = data.get("episodes", 1)

    valid, players = validate_players(players)
    if not valid:
        return jsonify({"error": players}), 400

    if PlayerType.HUMAN.value in players.values():
        return jsonify({"error": 'Human player not allowed in training'}), 400

    if not validate_env(env_name):
        return jsonify({"error": f'Invalid environment used for training choose one of {ENVIRONMENTS.keys()}'}), 400

    app.logger.info(
        f"{sid}: {user} training env={env_name}, with '{players}'")

    env_config: EnvironmentConfig = ENVIRONMENTS[env_name]
    env = env_config.env()

    env.reset()
    agent_iter = iter(env.agent_iter())
    current_agent = next(agent_iter)
    num_actions = env_config.num_actions
    obs_shape = env_config.observation_space
    session_state: SessionState = SessionState(
        env_config=env_config,
        env=env
    )
    populate_session_agents(session_state, env_config, players)
    ml_service.load_model(sid, session_state,
                          obs_shape, num_actions)
    session_state.agent_iter = agent_iter
    session_state.current_agent = session_state.agents[current_agent]

    session_state.runner = SessionRunner(
        sid, session_state, None)
    session_state.runner.start_training(episodes)
    with client_sessions_lock:
        client_sessions[sid] = session_state
    return {"success": True, "message": "Training started"}, 200


@app.route('/stop_training', methods=['POST'])
@token_required_api(roles=['Admin'])
def stop_training():
    sid = request.user['sub']
    with client_sessions_lock:
        session = client_sessions.get(sid, None)

    if not session:
        return {"success": False, "message": "No training session found"}, 404

    try:
        session.runner.stop()
        app.logger.info(f"{sid}: Training stopped and session cleaned.")
        return jsonify({"success": True, "message": "Training stopped"}), 200
    except Exception as e:
        app.logger.error(f"{sid}: Error stopping training: {e}")
        return jsonify({"success": False, "message": "Error stopping training"}), 500
