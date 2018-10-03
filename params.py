import os, sys
import models
import tensorflow as tf



# ------------------------------------
# ------------ ANN MODEL -------------
# ------------------------------------
#
MODEL_FN = models.lstm_model_fn
#MODEL_FN = models.cnn_model_fn

PRINT_SHAPE = False
# ------------------------------------
# ----- SETUP TRAINING---------------- export CUDA_VISIBLE_DEVICES=
# ------------------------------------
RESUME_TRAINING = False
MULTI_THREADING = True
TRAIN_SIZE = 76388 #76388 #54045 #317890 #159598 #1429485
NUM_EPOCHS = 7
BATCH_SIZE = 128
EVAL_AFTER_SEC = 120
TOTAL_STEPS = int((TRAIN_SIZE/BATCH_SIZE)*NUM_EPOCHS)

# ------------------------------------
# ------------- METADATA -------------
# ------------------------------------
PAD_WORD = '#=KS=#'
HEADER = ['sentence', 'class']
HEADER_DEFAULTS = [['NA'], ['NA']]
TEXT_FEATURE_NAME = 'sentence'
TARGET_NAME = 'class'
TARGET_LABELS = ['0', '1']
#WEIGHT_COLUNM_NAME = 'weight'

# ------------------------------------
# ----------- GLOVE EMBEDDING --------
# ------------------------------------
GLOVE_ACTIVE=True
TRAINABLE_GLOVE=False
MAX_DOCUMENT_LENGTH = 20
EMBEDDING_SIZE = 300 if GLOVE_ACTIVE else 30

# ------------------------------------
# ------------- TRAINING PARAMS ------------
# ------------------------------------
LEARNING_RATE = 0.01
# For LSTM0
FORGET_BIAS = 1.0
# For LSTM0
DROPOUT_RATE = 0.2
# For LSTM it refers to the size of the Cell, for CNN model instead are the FC layers
HIDDEN_UNITS = [64,32,16] #[96, 64, 16], None
# For CNN, kernel size
WINDOW_SIZE = 3
# For CNN, number of filters (i.e. feature maps)
FILTERS = 32

# ------------------------------------
# ------------- MODEL DIR ------------
# ------------------------------------
MODEL_NAME = str(sys.argv[1])
MODEL_DIR = os.path.join(os.getcwd(),'trained_models/{}'.format(MODEL_NAME))
INFERENCE_DIR = os.path.join(os.getcwd(), 'inference_results/')

# ------------------------------------
# ------- TRAIN & VALID PATH ---------
# ------------------------------------
# TRAIN_DATA_FILES_PATTERN = os.path.join(os.getcwd(),'data/train-data-maxlength16-subtitles.tsv')
# VALID_DATA_FILES_PATTERN = os.path.join(os.getcwd(),'data/valid-data-maxlength16-subtitles.tsv')
# VOCAB_LIST_FILE = os.path.join(os.getcwd(),'vocab/vocab_list_5k_2k_mystop_nodgts.tsv')
# N_WORDS_FILE = os.path.join(os.getcwd(),'vocab/n_words_5k_2k_mystop_nodgts.tsv')
TRAIN_DATA_FILES_PATTERN = os.path.join(os.getcwd(),'data_six/train_data_length3-16_v1.tsv')
VALID_DATA_FILES_PATTERN = os.path.join(os.getcwd(),'data_six/valid_data_length3-16_v1.tsv')
VOCAB_LIST_FILE = os.path.join(os.getcwd(),'data_six/vocab_list_5k_v1.tsv')
N_WORDS_FILE = os.path.join(os.getcwd(),'data_six/n_words_5k_v1.tsv')
with open(N_WORDS_FILE) as file:
    N_WORDS = int(file.read())+2
