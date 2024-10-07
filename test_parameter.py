TEST_N_AGENTS = 4

EXPLORATION = True  # True: unknown map, False: known map

GROUP_START = True  # True: start from the same location, False: start from different locations
INPUT_DIM = 8
EMBEDDING_DIM = 128
MAX_EPISODE_STEP = 128
UNBOUND_SPEED = False  # evader speed

USE_GPU = False
NUM_GPU = 0
NUM_META_AGENT = 1  # the number of processes
FOLDER_NAME = 'viper'
model_path = f'model/{FOLDER_NAME}'
gifs_path = f'results/gifs'

NUM_TEST = 1
SAVE_GIFS = False
SAVE_CSV = False
