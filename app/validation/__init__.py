from app.config.ml_env_config import ENVIRONMENTS
from app.config.player_state import PlayerType


def validate_env(env_name):
    return env_name in ENVIRONMENTS


def validate_players(players):
    if not players:
        return False, "Missing players"

    if isinstance(players, str):
        try:
            import json
            players = json.loads(players)
        except ValueError:
            return False, "Invalid players format"

    if not isinstance(players, dict):
        return False, "Players must be a dictionary"

    return True, players


def validate_env_players_comb(env_name, allowed_types, players):
    valid, players_val_response = validate_players(players)

    if not valid:
        return False, players_val_response

    if not validate_env(env_name):
        return False, "Invalid environment"

    env_cfg = ENVIRONMENTS[env_name]

    if len(players_val_response) != len(env_cfg.agents):
        return False, "Player count mismatch"

    human_seen = False
    for agent in env_cfg.agents:
        if agent not in players_val_response:
            return False, f"Missing agent: {agent}"
        agent_type = players_val_response[agent]
        if agent_type not in allowed_types:
            return False, f"Invalid type for {agent}: {agent_type}"
        if agent_type == PlayerType.HUMAN:
            if human_seen:
                return False, "More than one human player not supported atm"
            human_seen = True

    return True, players_val_response
