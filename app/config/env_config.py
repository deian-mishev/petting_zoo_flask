import os
from dotenv import load_dotenv
import eventlet

load_dotenv()

FLASK_KEY = os.getenv("FLASK_KEY")
FLASK_RUN_PORT= os.getenv("FLASK_RUN_PORT")
AGENT_VID_HEIGHT = os.getenv("AGENT_VID_HEIGHT")
AGENT_VID_WIDTH = os.getenv("AGENT_VID_WIDTH")

ATTARI_PRO_MODEL=os.getenv("ATTARI_PRO_MODEL")
ATTARI_PRO_WEIGHTS_PATH=os.getenv("ATTARI_PRO_WEIGHTS_PATH")
ATTARI_PRO_LOCK = eventlet.semaphore.Semaphore()

SEED = 0                        # seed for the pseudo-random number generator.
MINIBATCH_SIZE = 128            # mini-batch size.
TAU = 1e-3                      # soft update parameter.
GAMMA = 0.997                   # discount factor
ALPHA = 1e-3                    # learning rate
MEMORY_SIZE = 10_000            # size of memory buffer
EPSILON = 1.0             # initial rate of random modle choices
E_DECAY = 0.95            # e-decay rate for the e-greedy policy.
E_GROW = 1.001            # e-grow rate for the e-greedy policy.
E_MIN = 0.01              # minimum ε value for the e-greedy policy.
E_MAX = 1.0               # max ε value for the e-greedy policy.

INACTIVITY_TO = -1        # reward equals to O inactivity is punished to 
REWARD_POS_FACTOR = 5     # positive reward multiplication factor
REWARD_NEG_FACTO = 3      # negative reward multiplication factor

INPUT_TIMEOUT = 0.007