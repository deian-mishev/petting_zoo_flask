from typing import Optional
import eventlet
from pettingzoo.atari import boxing_v2, space_invaders_v2, tennis_v3, double_dunk_v3, ice_hockey_v2, mario_bros_v3, pong_v3, wizard_of_wor_v3
from dataclasses import dataclass, field
from typing import Optional, Callable
from app.config.env_config import EPSILON

@dataclass
class EnvironmentConfig:
    name: str
    KEY_MAP: dict
    model_path: str
    weights_path: str
    num_actions: int
    agents: list[str]
    observation_space: tuple[int, ...]
    epsilon: float = EPSILON
    env: Optional[Callable] = field(default=None)
    lock: eventlet.semaphore.Semaphore = field(
        default_factory=eventlet.semaphore.Semaphore)
    
ENVIRONMENTS = {
    'Boxing': EnvironmentConfig(
        name='boxing_v2',
        KEY_MAP={
            "0": 0, "s": 1, "8": 2, "6": 3, "4": 4, "2": 5,
            "9": 6, "7": 7, "3": 8, "1": 9, "w": 10, "d": 11,
            "a": 12, "x": 13, "e": 14, "q": 15, "c": 16, "z": 17
        },
        model_path="./resources/models/keras/boxing_v2.keras",
        weights_path="./resources/models/keras/boxing_v2.weights.h5",
        env=lambda render_mode="rgb_array": boxing_v2.env(
            render_mode=render_mode),
        num_actions=18,
        agents=['first_0', 'second_0'],
        observation_space=(52, 40, 1)
        # observation_space=(210, 160, 3)
    ),
    "Tennis": EnvironmentConfig(
        name='tennis_v3',
        KEY_MAP={
            "8": 2, "6": 3, "4": 4,
            "2": 5, "9": 6, "7": 7,
            "3": 8, "1": 9, "w": 10,
            "d": 11, "a": 12, "x": 13,
            "e": 14, "q": 15, "c": 16,
            "z": 17, "s": 1, "0": 0
        },
        model_path="./resources/models/keras/tennis_v3.keras",
        weights_path="./resources/models/keras/tennis_v3_weights.weights.h5",
        env=lambda render_mode="rgb_array": tennis_v3.env(
            render_mode=render_mode),
        num_actions=18,
        agents=['first_0', 'second_0'],
        observation_space=(52, 40, 1)
        # observation_space=(210, 160, 3)
    ),
    'Wizard of War': EnvironmentConfig(
        name='wizard_of_wor_v3',
        KEY_MAP={
            "8": 1, "6": 2, "4": 3,
            "2": 4, "9": 5, "7": 6,
            "3": 7, "1": 8, "s": 0
        },
        model_path="./resources/models/keras/wizard_of_wor_v3.keras",
        weights_path="./resources/models/keras/wizard_of_wor_v3_weights.weights.h5",
        env=lambda render_mode="rgb_array": wizard_of_wor_v3.env(
            render_mode=render_mode),
        num_actions=9,
        agents=['first_0', 'second_0'],
        observation_space=(52, 40, 1)
        # observation_space=(210, 160, 3)
    ),
    'Mario Bros': EnvironmentConfig(
        name='mario_bros_v3',
        KEY_MAP={
            "8": 2, "6": 3, "4": 4,
            "2": 5, "9": 6, "7": 7,
            "3": 8, "1": 9, "w": 10,
            "d": 11, "a": 12, "x": 13,
            "e": 14, "q": 15, "c": 16,
            "z": 17, "s": 1, "0": 0
        },
        model_path="./resources/models/keras/mario_bros_v3.keras",
        weights_path="./resources/models/keras/mario_bros_v3_weights.weights.h5",
        env=lambda render_mode="rgb_array": mario_bros_v3.env(
            render_mode=render_mode),
        num_actions=18,
        agents=['first_0', 'second_0'],
        observation_space=(52, 40, 1)
        # observation_space=(210, 160, 3)
    ),
    'Ice Hockey': EnvironmentConfig(
        name='ice_hockey_v2',
        KEY_MAP={
            "8": 2, "6": 3, "4": 4,
            "2": 5, "9": 6, "7": 7,
            "3": 8, "1": 9, "w": 10,
            "d": 11, "a": 12, "x": 13,
            "e": 14, "q": 15, "c": 16,
            "z": 17, "s": 1, "0": 0
        },
        model_path="./resources/models/keras/ice_hockey_v2.keras",
        weights_path="./resources/models/keras/ice_hockey_v2_weights.weights.h5",
        env=lambda render_mode="rgb_array": ice_hockey_v2.env(
            render_mode=render_mode),
        num_actions=18,
        agents=['first_0', 'second_0'],
        observation_space=(52, 40, 1)
        # observation_space=(210, 160, 3)
    ),
    'Double Dunk': EnvironmentConfig(
        name='double_dunk_v3',
        KEY_MAP={
            "8": 2, "6": 3, "4": 4,
            "2": 5, "9": 6, "7": 7,
            "3": 8, "1": 9, "w": 10,
            "d": 11, "a": 12, "x": 13,
            "e": 14, "q": 15, "c": 16,
            "z": 17, "s": 1, "0": 0
        },
        model_path="./resources/models/keras/double_dunk_v3.keras",
        weights_path="./resources/models/keras/double_dunk_v3_weights.weights.h5",
        env=lambda render_mode="rgb_array": double_dunk_v3.env(
            render_mode=render_mode),
        num_actions=18,
        agents=['first_0', 'second_0'],
        observation_space=(52, 40, 1)
        # observation_space=(210, 160, 3)
    ),
    'Pong': EnvironmentConfig(
        name='pong_v3_model',
        KEY_MAP={
            "4": 3, "6": 2, "s": 1, "0": 0
        },
        model_path="./resources/models/keras/pong_v3_model.keras",
        weights_path="./resources/models/keras/pong_v3_weights.weights.h5",
        env=lambda render_mode="rgb_array": pong_v3.env(
            render_mode=render_mode),
        num_actions=6,
        agents=['first_0', 'second_0'],
        observation_space=(52, 40, 1)
        # observation_space=(210, 160, 3)
    ),
    "Space Invaders": EnvironmentConfig(
        name='space_invaders_v2',
        KEY_MAP={
            "8": 2, "6": 4, "4": 3, "2": 5, "s": 1,  "0": 0
        },
        model_path="./resources/models/keras/space_invaders_v2.keras",
        weights_path="./resources/models/keras/space_invaders_v2_weights.weights.h5",
        env=lambda render_mode="rgb_array": space_invaders_v2.env(
            render_mode=render_mode),
        num_actions=6,
        agents=['first_0', 'second_0'],
        observation_space=(52, 40, 1)
        # observation_space=(210, 160, 3)
    )
}

ATTARI_PRO_MODEL_HEADS={cfg.name: cfg.num_actions for cfg in ENVIRONMENTS.values()}