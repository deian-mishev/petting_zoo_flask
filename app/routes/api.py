from flask_socketio import disconnect
from flask import request, jsonify, session, render_template

from app import app, socketio, client_sessions, client_sessions_lock
from app.services.ml_service import ml_service
from app.services.session_runner import SessionRunner
from app.config.session_state import SessionState
from app.config.ml_env_config import EnvironmentConfig, ENVIRONMENTS
from app.config.env_config import *
from app.config.oauth2_config import login_required, roles_required
from app.validation import validate_env_players_comb
from app.services.session import cleanup_session, get_available_environments_and_nemesis, populate_session_agents

@app.route('/preconnect', methods=['GET'])
@login_required
@roles_required('User')
def preconnect():
    envs, players = get_available_environments_and_nemesis()
    return jsonify({
        "environments": envs,
        "ai_players":  players
    })


@socketio.on('connect')
def on_connect():
    user = session.get('user')
    roles = session.get('roles', [])

    if not user or 'User' not in roles:
        disconnect()
        return

    sid = request.sid
    env_name = request.args.get("env")
    players = request.args.get("players")

    _, available_players = get_available_environments_and_nemesis()
    valid, players = validate_env_players_comb(
        env_name, available_players, players)
    if not valid:
        disconnect()
        return False, players

    app.logger.info(
        f"{sid}: {user['name']} connected with roles {roles} in env={env_name}, facing '{players}'")

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

    with client_sessions_lock:
        client_sessions[sid] = session_state

    session_state.runner = SessionRunner(
        sid, session_state, socketio)
    session_state.runner.start()


@socketio.on('input')
def on_input(keys: list[str]):
    sid = request.sid
    with client_sessions_lock:
        session: SessionState = client_sessions.get(sid)
        if session is not None:
            with session.lock:
                for key in keys:
                    action = session.env_config.KEY_MAP.get(key)
                    if action is not None:
                        session.next_human_action = action
                        break
                else:
                    session.next_human_action = 0


@socketio.on('disconnect')
def on_disconnect():
    sid = request.sid
    cleanup_session(sid)
    app.logger.info(f"{sid}: User disconnected...")


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
@login_required
def index(path):
    return render_template("index.html", width=AGENT_VID_WIDTH, height=AGENT_VID_HEIGHT)
