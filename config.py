AUDIO_PATH = "in_audio"
TARGET_LANGUAGE = "en"
DATASET_PATH = "dataset"
WHISPER_MODEL = "distil-large-v3"
WHISPER_FORCE_CPU = True
WHISPER_THREADS = 16 # 0 uses 4 by default in whisper

TRAIN_EPOCHS = 6 # Default 10
# Note: we recommend that BATCH_SIZE * GRAD_ACUMM_STEPS need to be at least 252 for more 
# efficient training. You can increase/decrease BATCH_SIZE but then set GRAD_ACUMM_STEPS accordingly.
TRAIN_BATCH_SIZE = 2 #default 4
TRAIN_GRAD_ACCUMULATION_STEPS = 50 #default 1
TRAIN_MAX_PERMIT_AUDIO_S = 11 # default 11
TRAIN_TRAINING_PATH = "training"
TRAIN_OUTPUT_PATH = "model"