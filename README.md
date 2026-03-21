# PETTING-ZOO-FLASK

## Description

A multi-agent reinforcement learning game platform built with Flask and Socket.IO. Human players can compete against AI agents in classic Atari games in real time, while administrators can trigger autonomous training sessions. AI agents learn via Deep Q-Networks (DQN) with experience replay. A multi-head "Atari Pro" model generalises across all supported environments using a shared convolutional backbone and per-environment output heads. Authentication is handled through OAuth2/OpenID Connect (Keycloak) with role-based access control.

## Technologies

PETTING-ZOO-FLASK uses the following open source libraries:

* [PYTHON3] - Programming language
* [FLASK] - Lightweight Python web framework
* [FLASK_SOCKETIO] - Real-time bi-directional communication for Flask apps
* [TENSORFLOW] - Machine learning library
* [KERAS] - High-level API for neural networks
* [NUMPY] - Numerical computing library
* [PETTINGZOO] - Multi-agent reinforcement learning environments
* [MULTI_AGENT_ALE] - Atari Learning Environment for multi-agent play
* [AUTOROM] - Atari ROM management utility
* [GYMNASIUM] - Modernized OpenAI Gym interface for RL environments
* [OPENCV] - Image processing (grayscale conversion, frame resizing)
* [PYMONGO] - MongoDB driver for experience replay persistence
* [AUTHLIB] - OAuth2 / OpenID Connect client integration
* [PYTHON_JOSE] - JWT token validation
* [EVENTLET] - Coroutine-based async networking
* [PYTHON_DOTENV] - Environment variable loading from `.env` files
* [PYNGROK] - ngrok tunnel support for remote access

## Supported Environments

| Environment       | PettingZoo Module       | Actions | Observation   |
|-------------------|-------------------------|---------|---------------|
| Boxing            | `boxing_v2`             | 18      | (52, 40, 1)   |
| Tennis            | `tennis_v3`             | 18      | (52, 40, 1)   |
| Wizard of Wor     | `wizard_of_wor_v3`      | 9       | (52, 40, 1)   |
| Mario Bros        | `mario_bros_v3`         | 18      | (52, 40, 1)   |
| Ice Hockey        | `ice_hockey_v2`         | 18      | (52, 40, 1)   |
| Double Dunk       | `double_dunk_v3`        | 18      | (52, 40, 1)   |
| Pong              | `pong_v3`               | 6       | (52, 40, 1)   |
| Space Invaders    | `space_invaders_v2`     | 6       | (52, 40, 1)   |

## Player Types

| Type          | Enum value      | Description                                                                 |
|---------------|-----------------|-----------------------------------------------------------------------------|
| `HUMAN`       | `human`         | Human player controlled via keyboard input over Socket.IO                   |
| `COMPUTER`    | `ai_regular`    | Single-environment DQN agent with a dedicated model per game                |
| `ATARI_PRO`   | `atari_ai_pro`  | Multi-head DQN agent sharing a common backbone across all environments (singleton — only one active at a time) |

## ML Architecture

All agents use a Deep Q-Network trained with experience replay.

**Preprocessing pipeline:**
```
RGB frame → Grayscale → Resize to (52, 40) → Normalize [0, 1] → shape (52, 40, 1)
```

**Network (single-head):**
```
Input (52, 40, 1)
→ Rescaling (÷255)
→ Conv2D(32, 8×8, stride=4) + ReLU
→ Conv2D(64, 4×4, stride=2) + ReLU
→ Conv2D(64, 3×3, stride=1) + ReLU
→ Flatten
→ Dense(512) + ReLU
→ Dense(num_actions)
```

**Multi-head (ATARI_PRO):** the shared Conv + Dense(512) backbone is reused; each environment gets its own `Dense(num_actions)` output head.

**Training algorithm:**
```
Target:  y = r + (1 - done) × γ × max Q'(s', a')
Loss:    MSE(y, Q(s, a))
Update:  w_target = τ × w + (1 - τ) × w_target   (soft update)
```

**Default hyperparameters:**

| Parameter        | Value    | Description                          |
|------------------|----------|--------------------------------------|
| `GAMMA`          | `0.997`  | Discount factor                      |
| `ALPHA`          | `1e-3`   | Adam learning rate                   |
| `TAU`            | `1e-3`   | Soft update coefficient              |
| `MINIBATCH_SIZE` | `128`    | Experience replay batch size         |
| `MEMORY_SIZE`    | `10 000` | Max experiences per agent buffer     |
| `EPSILON`        | `1.0`    | Initial exploration rate             |
| `E_DECAY`        | `0.95`   | Epsilon decay multiplier per step    |
| `E_MIN`          | `0.01`   | Minimum exploration floor            |
| `REWARD_POS_FACTOR` | `5`   | Multiplier applied to positive rewards |
| `REWARD_NEG_FACTO`  | `3`   | Multiplier applied to negative rewards |
| `INACTIVITY_TO`  | `-1`     | Penalty for zero-reward steps        |

## Installation

Install required Python packages using pip:

```sh
pip install -r requirements.txt
```

Install Atari ROMs (required by PettingZoo Atari environments):

```sh
AutoROM --accept-license
```

## Environment Variables

Copy `test_run/.env.example` to `.env` in the project root and fill in your values:

```env
# Flask
FLASK_KEY=change-me-in-production
FLASK_RUN_PORT=5000

# Rendering
AGENT_VID_WIDTH=160
AGENT_VID_HEIGHT=210

# MongoDB
MONGO_DB_URI=mongodb://localhost:27017/rl_db

# OAuth2 / Keycloak
OAUTH2_SERVER_URL=http://localhost:8080
OAUTH2_REALM=petting-zoo
OAUTH2_CLIENT_ID=petting-zoo-client
OAUTH2_CLIENT_SECRET=dev-secret-change-in-prod

# Model paths (Atari Pro multi-head model)
ATTARI_PRO_MODEL=./resources/models/keras/atari_pro.keras
ATTARI_PRO_WEIGHTS_PATH=./resources/models/keras/atari_pro.weights.h5
```

Per-environment models are saved automatically to `./resources/models/keras/` on training. The `resources/` directory is git-ignored.

## Running and Building

* **WITH CLI COMMANDS**

    Install dependencies and ROMs, then run:

    ```sh
    pip install -r requirements.txt
    AutoROM --accept-license
    python run.py
    ```

    The server starts on the port defined by `FLASK_RUN_PORT` (default `5000`).
    MongoDB and Keycloak must be running and reachable at the URIs in `.env`.

* **WITH DOCKER COMPOSE (recommended for local dev)**

    All services (app, MongoDB, Keycloak) are defined in `test_run/docker-compose.yml`.

    First-time setup:

    ```sh
    cp test_run/.env.example .env
    # edit .env as needed
    cd test_run
    docker compose up --build
    ```

    Subsequent runs (no code changes):

    ```sh
    cd test_run
    docker compose up
    ```

    **Services:**

    | Service   | URL                        | Notes                              |
    |-----------|----------------------------|------------------------------------|
    | App       | http://localhost:5000      | Flask + Socket.IO                  |
    | Keycloak  | http://localhost:8080      | Admin console: `admin / admin`     |
    | MongoDB   | mongodb://localhost:27017  | Database: `rl_db`                  |

    Trained models are persisted to `./resources/` on the host via a Docker volume mount and survive container restarts.

* **Keycloak — pre-seeded configuration**

    The realm import at `test_run/keycloak/realm-export.json` seeds the following on first boot:

    | Item            | Value                        |
    |-----------------|------------------------------|
    | Realm           | `petting-zoo`                |
    | Client ID       | `petting-zoo-client`         |
    | Client secret   | `dev-secret-change-in-prod`  |
    | Redirect URI    | `http://localhost:5000/*`    |
    | Roles           | `User`, `Admin`              |
    | Demo users      | `admin / admin123` (User + Admin), `player / player123` (User) |

## API Reference

### WebSocket Events (Socket.IO)

| Event        | Direction       | Auth                          | Description                                                     |
|--------------|-----------------|-------------------------------|-----------------------------------------------------------------|
| `connect`    | client → server | `login_required`, role `User` | Opens a game session; payload selects environment and opponents |
| `input`      | client → server | session                       | Sends keyboard key(s) for the human player's current turn       |
| `disconnect` | client → server | session                       | Tears down the session, saves model, persists experiences       |
| `frame`      | server → client | —                             | Emits a base64-encoded PNG of the current game frame            |

### HTTP Endpoints

| Route          | Method | Auth                          | Description                                          |
|----------------|--------|-------------------------------|------------------------------------------------------|
| `/`            | GET    | `login_required`              | Serves the main SPA (`index.html`)                   |
| `/preconnect`  | GET    | `login_required`, role `User` | Returns available environments and opponent options  |
| `/login`       | GET    | —                             | Initiates OAuth2 login redirect                      |
| `/auth`        | GET    | —                             | OAuth2 callback; exchanges code for tokens           |
| `/logout`      | GET    | session                       | Clears session and redirects to Keycloak logout      |

### Admin Endpoints (Bearer token required, role `Admin`)

| Route            | Method | Description                                              |
|------------------|--------|----------------------------------------------------------|
| `/train`         | POST   | Starts an autonomous training session (no human player)  |
| `/stop_training` | POST   | Stops the running training session                       |

**`POST /train` payload:**
```json
{
  "env": "boxing_v2",
  "players": ["atari_ai_pro", "ai_regular"],
  "max_episodes": 500
}
```

## Project Structure

```
petting_zoo_flask/
├── run.py                        # Entry point — registers blueprints, starts server
├── Dockerfile
├── requirements.txt
├── .env                          # git-ignored — copy from test_run/.env.example
├── .gitignore
├── wsl-dev-deps.sh               # WSL dependency helper script
├── resources/                    # git-ignored — trained models saved here
│   └── models/keras/
├── test_run/                     # Docker Compose dev environment
│   ├── docker-compose.yml
│   ├── .env.example
│   └── keycloak/
│       └── realm-export.json     # Pre-seeded realm, client, roles, and users
└── app/
    ├── __init__.py               # Flask app, SocketIO, and global session store
    ├── config/
    │   ├── env_config.py         # All env vars, hyperparameters, model paths
    │   ├── ml_env_config.py      # Per-environment configs (actions, agents, model heads)
    │   ├── session_state.py      # SessionState dataclass
    │   ├── player_state.py       # PlayerState, Experience, PlayerType
    │   ├── logging_config.py     # Rotating file / stdout logger setup
    │   ├── oauth2_config.py      # Keycloak OAuth2 + JWT decorators
    │   └── persistance_config.py # MongoDB connection and collections
    ├── routes/
    │   ├── api.py                # Socket.IO events + HTTP game routes
    │   └── admin.py              # Admin training endpoints
    ├── services/
    │   ├── ml_service.py         # DQN model build, train, inference (singleton)
    │   ├── session_runner.py     # Eventlet game loop and training loop
    │   ├── session.py            # Session lifecycle and agent population
    │   ├── experience_store.py   # MongoDB experience replay CRUD (singleton)
    │   └── rendering_service.py  # Frame → base64 PNG
    ├── validation/
    │   └── __init__.py           # Request validation helpers
    ├── static/
    │   ├── images/
    │   └── js/
    │       └── agent-client.js   # Socket.IO game client
    └── templates/
        └── index.html            # SPA entry point
```

## Concurrency Model

The server runs on **eventlet** greenlets (cooperative multitasking):

- Each active game session runs in its own greenlet spawned by `SessionRunner`.
- `eventlet.semaphore.Semaphore` guards the global session map, each session, and each model.
- The ATARI_PRO model is a singleton protected by a global `ATTARI_PRO_LOCK`; only one session may use it at a time.
- Frame delivery targets ~143 FPS (`INPUT_TIMEOUT = 0.007 s`) using `socketio.sleep()` for cooperative yielding.

## Logging

| Environment | Output                                      |
|-------------|---------------------------------------------|
| `DEV`       | stdout                                      |
| Production  | Rotating file (5 MB per file, 5 backups)    |

Set the `LOG_LEVEL` and `ENV` variables in `.env` to control behaviour.

[PYTHON3]: https://www.python.org/downloads/release/python-3/
[FLASK]: https://flask.palletsprojects.com/
[FLASK_SOCKETIO]: https://flask-socketio.readthedocs.io/
[TENSORFLOW]: https://www.tensorflow.org/
[KERAS]: https://keras.io/
[NUMPY]: https://numpy.org/
[PETTINGZOO]: https://www.pettingzoo.ml/
[MULTI_AGENT_ALE]: https://github.com/Farama-Foundation/Multi-Agent-ALE
[AUTOROM]: https://github.com/Farama-Foundation/AutoROM
[GYMNASIUM]: https://gymnasium.farama.org/
[OPENCV]: https://opencv.org/
[PYMONGO]: https://pymongo.readthedocs.io/
[AUTHLIB]: https://authlib.org/
[PYTHON_JOSE]: https://python-jose.readthedocs.io/
[EVENTLET]: https://eventlet.net/
[PYTHON_DOTENV]: https://pypi.org/project/python-dotenv/
[PYNGROK]: https://pyngrok.readthedocs.io/
