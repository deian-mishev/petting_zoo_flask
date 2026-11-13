# PETTING-ZOO-FLASK

## Description

A multi-agent reinforcement learning game platform built with Flask and Socket.IO. Human players can compete against AI agents in classic Atari games in real time, while administrators can trigger autonomous training sessions. AI agents learn via Deep Q-Networks (DQN) with experience replay. A multi-head "Atari Pro" model generalises across all supported environments using a shared convolutional backbone and per-environment output heads. Authentication is handled through OAuth2/OpenID Connect (Keycloak) with role-based access control.

## Technologies

* [PYTHON3] - Programming language
* [FLASK] - Lightweight Python web framework
* [FLASK_SOCKETIO] - Real-time bi-directional communication (threading async mode)
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

| Type        | Enum value     | Description                                                                                              |
|-------------|----------------|----------------------------------------------------------------------------------------------------------|
| `HUMAN`     | `human`        | Human player controlled via keyboard input over Socket.IO                                                |
| `ATARI_PRO` | `atari_ai_pro` | Multi-head DQN agent sharing a common backbone across all environments (singleton — one active at a time) |

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

| Parameter           | Value    | Description                            |
|---------------------|----------|----------------------------------------|
| `GAMMA`             | `0.997`  | Discount factor                        |
| `ALPHA`             | `1e-3`   | Adam learning rate                     |
| `TAU`               | `1e-3`   | Soft update coefficient                |
| `MINIBATCH_SIZE`    | `128`    | Experience replay batch size           |
| `MEMORY_SIZE`       | `10 000` | Max experiences per agent buffer       |
| `EPSILON`           | `1.0`    | Initial exploration rate               |
| `E_DECAY`           | `0.95`   | Epsilon decay multiplier per step      |
| `E_MIN`             | `0.01`   | Minimum exploration floor              |
| `REWARD_POS_FACTOR` | `5`      | Multiplier applied to positive rewards |
| `REWARD_NEG_FACTO`  | `3`      | Multiplier applied to negative rewards |
| `INACTIVITY_TO`     | `-1`     | Penalty for zero-reward steps          |

## Installation

```sh
pip install -r requirements.txt
AutoROM --accept-license
```

## Environment Variables

Copy `.env.example` to `.env` in the project root and fill in your values:

```env
# Flask
FLASK_KEY=change-me-in-production
FLASK_RUN_PORT=5001

# Video rendering
AGENT_VID_WIDTH=480
AGENT_VID_HEIGHT=630

# MongoDB (Docker port mapped to localhost)
MONGO_ENABLED=false
MONGO_DB_URI=mongodb://localhost:27017/rl_db

# Keycloak (Docker port mapped to localhost)
OAUTH2_SERVER_URL=http://localhost:8080
OAUTH2_REALM=petting-zoo
OAUTH2_CLIENT_ID=petting-zoo-client
OAUTH2_CLIENT_SECRET=dev-secret-change-in-prod

# Atari Pro multi-head model (optional — leave blank to skip loading)
ATTARI_PRO_MODEL=
ATTARI_PRO_WEIGHTS_PATH=
```

### Model checkpoints

`MODEL_CHECKPOINT_BACKEND=local` is the Linux/WSL development default, for local
disks only. It stores paired model files under `resources/models/keras/` with
POSIX shared-reader/exclusive-writer file locks. Readers do not block other
readers; publication and cleanup wait for readers to finish. SQLite is no longer
used by checkpoint code. Existing SQLite files are left untouched. Stop all
older processes before using the new local backend: the old and new locks do
not coordinate with each other. Do not use local locks on network shares.

Azure Container Apps must explicitly use `MODEL_CHECKPOINT_BACKEND=azure_blob`.
When `CONTAINER_APP_NAME` is present, local load/save is rejected before writing
checkpoint files. Missing Blob configuration never silently falls back to the
mounted file share. The earlier SQLite errors demonstrate a locking failure,
not that the existing Azure Files share or model files are corrupt.

The opt-in `azure_blob` backend stores each model and target-weights pair in an
immutable, checksummed Blob. A conditional manifest update publishes one pair
at a time, with sequential version numbers. The current version and up to
three earlier rollback targets are retained. Cleanup is retried after later
saves if deletion fails. An uploaded candidate rejected before any publication
attempt is deleted best-effort. A lost publication response is reconciled only
when the manifest's generation and current entry exactly match that attempt.
Uploads with an unverified publication outcome are deliberately not automatically
deleted; monitor Blob usage. Thus retention limits published history, not total
storage usage after failures. Concurrent
sessions can play, but if another session publishes first, the later session's
training update is discarded rather than overwriting the newer model. Storage
is contacted on model load and save, not for individual gameplay frames;
loading, training, validation and network transfers can still affect resources.

Use your **existing storage account**: create a private Blob container alongside
the existing file share, not another storage account or a folder inside the share.
The Blob container is accessed through the Azure SDK; it is not mounted as a
filesystem. No changes to the file-share folder layout are required.

To switch an existing deployment, **stop all old app instances and writers**
before migrating, including running jobs and older revisions. Provision a
Blob container and grant the app's managed identity **Storage Blob
Data Contributor** on that container (or equivalent read, write and delete
permissions). Confirm the container network and identity access. No Azure
resources are created or permissions assigned by the application.

Set these variables in the app and in the controlled migration environment:

```text
MODEL_CHECKPOINT_BACKEND=azure_blob
MODEL_CHECKPOINT_BLOB_ACCOUNT_URL=https://<account>.blob.core.windows.net
MODEL_CHECKPOINT_BLOB_CONTAINER=<container>
MODEL_CHECKPOINT_MANAGED_IDENTITY_CLIENT_ID=<user-assigned-identity-client-id>
```

Leave the client ID blank for a system-assigned identity. The migration
environment must have the same managed identity access, dependencies, mounted
local files and configured model paths as the app. Use exactly the same model
path spelling in every instance and in migration commands: its normalized path
identifies the logical Blob checkpoint. First list the actual configured paths
(including the optional Atari Pro / Human model), without contacting Blob storage:

```sh
python -m app.services.blob_checkpoints inventory
```

For each listed model, run, for example:

```sh
python -m app.services.blob_checkpoints bootstrap ./resources/models/keras/boxing_v2.keras ./resources/models/keras/boxing_v2.weights.h5 --writers-stopped
python -m app.services.blob_checkpoints status ./resources/models/keras/boxing_v2.keras
```

`--writers-stopped` is your explicit operational confirmation, not an automatic
lock or a mechanism that stops apps. Bootstrap reads the source files without
modifying them or creating backups. It imports the current local manifest and retained history when present,
or the existing legacy model/weights pair, validating each pair. A missing,
incomplete or invalid pair fails instead of silently creating a new model. For
an intentionally new model only, use `bootstrap <model-path> <weights-path>
--empty --writers-stopped`; it refuses existing local model files or checkpoint history. Blob
manifests cannot be overwritten by bootstrap. Repeat for all configured model
paths, including the Atari Pro path if configured, run `status-all` to check
that every configured model has a valid manifest, then
switch the app to `azure_blob`. A missing Blob manifest or unavailable storage
prevents that game from starting; there is no fallback to SQLite or new models.
Keep old writers stopped after cutover, including during deployment rollouts.

#### Deployment order

1. Build and push the new image without deploying: run the Azure pipeline with
   `deployApp=false`. This does not switch the running app's storage backend.
2. Create the private Blob container and assign the container-scoped role.
   Keep your existing share, model files, and mounted paths intact.
3. Arrange a maintenance window and stop every old app revision and training
   writer. Use a separate, manually triggered Azure Container Apps Job with the
   new image, the managed identity, and the existing share mounted **read-only**
   at `/app/resources` for migration. Do not use a serving replica as the
   migration environment. Give the job the variables listed above and the same
   configured model paths, including `ATTARI_PRO_MODEL` and its weights path.
4. Run `inventory`, then bootstrap each model, then `status-all` in that job.
   Stop if any command fails; do not enable the new backend over missing models.
   These commands require SDK dependencies and identity/network access, not SQLite.
5. Deploy the new image and the Blob variables together, then enable traffic
   only to the new revision. Confirm game startup and saved version progression.
   Do not reactivate old local writers as a rollback after Blob training begins.

The pipeline's deployment step refuses to update an app not yet configured for
`azure_blob`; the initial image/backend cutover is a deliberate maintenance
operation. After cutover, normal pipeline deployment can resume. No live Azure
integration test is performed by the local unit tests. An incorrect identity,
firewall, endpoint, or missing manifest will still prevent game startup, with a
recoverable client error; fix configuration rather than falling back to old models.

To roll back, use `status <model-path>` to list retained versions, then run
`rollback <model-path> <version>` with the same module. Rollback checks the
stored pair's checksum and conditionally switches the manifest; concurrent
changes make it fail instead of overwriting them. Previously running sessions
become stale. The next publication continues above the highest issued version.
Validate model behavior separately; rollback targets are not a substitute for
independent disaster recovery, and checksums do not prevent bad training or
unauthorized modifications.

## Running

**With Docker Compose (recommended for local dev)**

`local_test/docker-compose.yml` runs MongoDB and Keycloak in Docker with ports mapped to `localhost`. The Flask app runs directly on the host — this avoids OAuth2 redirect URL conflicts between the browser and the server.

First-time setup:

```sh
cp .env.example .env
# edit .env as needed
cd local_test && docker compose up -d
# wait for Keycloak to be healthy (~90s), then in the project root:
python run.py
```

Subsequent runs:

```sh
cd local_test && docker compose up -d && cd ..
python run.py
```

**Services:**

| Service   | URL                        | Notes                           |
|-----------|----------------------------|---------------------------------|
| Flask app | http://localhost:5001      | Run locally via `python run.py` |
| Keycloak  | http://localhost:8080      | Admin console: `admin / admin`  |
| MongoDB   | mongodb://localhost:27017  | Database: `rl_db`               |

**Keycloak — pre-seeded configuration**

The realm import at `local_test/keycloak/realm-export.json` seeds the following on first boot:

| Item          | Value                        |
|---------------|------------------------------|
| Realm         | `petting-zoo`                |
| Client ID     | `petting-zoo-client`         |
| Client secret | `dev-secret-change-in-prod`  |
| Redirect URI  | `http://localhost:5001/*`    |
| Roles         | `User`, `Admin`              |
| Demo users    | `admin / admin123` (User + Admin), `player / player123` (User) |

## API Reference

### WebSocket Events (Socket.IO)

| Event        | Direction       | Auth                          | Description                                                     |
|--------------|-----------------|-------------------------------|-----------------------------------------------------------------|
| `connect`    | client → server | `login_required`, role `User` | Opens a game session; payload selects environment and opponents |
| `input`      | client → server | session                       | Sends keyboard key(s) for the human player's current turn       |
| `disconnect` | client → server | session                       | Tears down the session, saves model, persists experiences       |
| `frame`      | server → client | —                             | Emits a binary JPEG of the current game frame                   |

### HTTP Endpoints

| Route         | Method | Auth                          | Description                                         |
|---------------|--------|-------------------------------|-----------------------------------------------------|
| `/`           | GET    | `login_required`              | Serves the main SPA (`index.html`)                  |
| `/preconnect` | GET    | `login_required`, role `User` | Returns available environments and opponent options |
| `/login`      | GET    | —                             | Initiates OAuth2 login redirect                     |
| `/auth`       | GET    | —                             | OAuth2 callback; exchanges code for tokens          |
| `/logout`     | GET    | session                       | Clears session and redirects to Keycloak logout     |

### Admin Endpoints (Bearer token required, role `Admin`)

| Route            | Method | Description                                             |
|------------------|--------|---------------------------------------------------------|
| `/train`         | POST   | Starts an autonomous training session (no human player) |
| `/stop_training` | POST   | Stops the running training session                      |

**`POST /train` payload:**
```json
{
  "env": "Boxing",
  "players": {"first_0": "atari_ai_pro", "second_0": "atari_ai_pro"},
  "episodes": 5
}
```

## Project Structure

```
petting_zoo_flask/
├── run.py                        # Entry point — starts server
├── requirements.txt
├── .env.example                  # Copy to .env and fill in values
├── .env                          # git-ignored
├── .gitignore
├── local_test/                   # Docker Compose dev environment
│   ├── docker-compose.yml
│   └── keycloak/
│       └── realm-export.json     # Pre-seeded realm, client, roles, and users
├── resources/                    # git-ignored — trained models saved here
│   └── models/keras/
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
    │   ├── session_runner.py     # Threading game loop and training loop
    │   ├── session.py            # Session lifecycle and agent population
    │   ├── experience_store.py   # MongoDB experience replay CRUD (singleton)
    │   └── rendering_service.py  # Frame → binary JPEG
    ├── validation/
    │   └── __init__.py           # Request validation helpers
    ├── static/
    │   └── js/
    │       └── agent-client.js   # Socket.IO game client
    └── templates/
        └── index.html            # SPA entry point
```

## Concurrency Model

The server runs Flask-SocketIO in **threading** mode:

- Each active game session runs in its own `threading.Thread` spawned by `SessionRunner`.
- A `threading.Lock` guards the global session map and each regular model's training and save operations; sessions and Atari Pro use Eventlet semaphores.
- Sessions can load separate model copies. Checkpoint publication protects file pairs from partial saves, but does not merge simultaneous training updates.
- Frame delivery targets ~143 FPS (`INPUT_TIMEOUT = 0.007 s`).

## Logging

| Environment | Output                                   |
|-------------|------------------------------------------|
| `DEV`       | stdout                                   |
| Production  | Rotating file (5 MB per file, 5 backups) |

Set `LOG_LEVEL` and `ENV` in `.env` to control behaviour.

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
[PYTHON_DOTENV]: https://pypi.org/project/python-dotenv/
[PYNGROK]: https://pyngrok.readthedocs.io/
