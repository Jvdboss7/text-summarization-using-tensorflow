import os 
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

# Data Ingestion constants
ARTIFACTS_DIR = os.path.join("artifacts", TIMESTAMP)
BUCKET_NAME = 'text-summarization-using-tensorflow'
ZIP_FILE_NAME = 'dataset.zip'
DATA_DIR = "data"
RAW_FILE_NAME = 'dataset'
DATA_INGESTION_ARTIFACTS_DIR = 'DataIngestionArtifacts'
DATA_INGESTION_TRAIN_DIR = 'train'
DATA_INGESTION_TEST_DIR = 'test'
DATA_INGESTION_VALID_DIR = 'validation'
DATASET_DICT_JSON_FILE_NAME = 'dataset_dict.json' 

# Data transformation constants
DATA_TRANSFORMATION_ARTIFACTS_DIR = 'DataTransformationArtifacts'
RAW_DATASET_NAME = 'raw_datasets'
MAX_INPUT_LENGTH = 1024
MAX_TARGET_LENGTH = 128
FAKE_PREDS = ["hello there", "general kenobi"]
FAKE_LABELS = ["hello there", "general kenobi"]

# Model trainer cosntants
MODEL_TRAINER_ARTIFACTS_DIR = 'ModelTrainerArtifacts'
TRAINED_MODEL_DIR = 'trained_model'
TRAINED_MODEL_NAME = 'tf_model.h5'
BATCH_SIZE = 1
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_TRAIN_EPOCHS = 1
EPOCHS = 1

# AWS CONSTANTS
AWS_ACCESS_KEY_ID_ENV_KEY = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV_KEY = "AWS_SECRET_ACCESS_KEY"
REGION_NAME = "ap-south-1"