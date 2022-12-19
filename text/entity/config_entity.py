from dataclasses import dataclass
from text.constants import *
import os

@dataclass
class DataIngestionConfig:
    def __init__(self):
        self.BUCKET_NAME: str = BUCKET_NAME
        self.ZIP_FILE_NAME:str = ZIP_FILE_NAME
        self.S3_DATA_DIR = DATA_DIR
        self.DATA_INGESTION_ARTIFACTS_DIR: str = os.path.join(os.getcwd(), ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS_DIR,DATA_DIR)
        self.TRAIN_DATA_ARTIFACT_DIR = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR, DATA_INGESTION_TRAIN_DIR)
        self.TEST_DATA_ARTIFACT_DIR = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR, DATA_INGESTION_TEST_DIR)
        self.VALID_DATA_ARTIFACT_DIR = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR, DATA_INGESTION_VALID_DIR)
        self.ZIP_FILE_DIR = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR)
        self.ZIP_FILE_PATH = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR, self.ZIP_FILE_NAME)
        self.UNZIPPED_FILE_PATH = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR, RAW_FILE_NAME)
        self.DATASET_DICT_JSON_FILE = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR,DATASET_DICT_JSON_FILE_NAME)
        self.ALL_DATASET_FILE_PATH = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR)
 
class DataTransformationConfig: 
    def __init__(self): 
        self.ROOT_DIR: str = os.path.join(os.getcwd(), ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS_DIR)
        self.DATA_TRANSFORMATION_ARTIFACTS_DIR: str = os.path.join(os.getcwd(),ARTIFACTS_DIR,DATA_TRANSFORMATION_ARTIFACTS_DIR)
        self.RAW_DATASET = os.path.join(self.DATA_TRANSFORMATION_ARTIFACTS_DIR)

@dataclass
class ModelTrainerConfig:
     def __init__(self):
        self.TRAINED_MODEL_DIR: str = os.path.join(os.getcwd(), ARTIFACTS_DIR, MODEL_TRAINER_ARTIFACTS_DIR,TRAINED_MODEL_DIR)
        self.TRAINED_MODEL_PATH = os.path.join(self.TRAINED_MODEL_DIR)
        
@dataclass
class ModelEvaluationConfig:
    def __init__(self):
        self.EVALUATED_MODEL_DIR: str = os.path.join(from_root(), ARTIFACTS_DIR, MODEL_EVALUATION_ARTIFACTS_DIR)
        self.EVALUATED_LOSS_CSV_PATH = os.path.join(self.EVALUATED_MODEL_DIR, MODEL_EVALUATION_FILE_NAME)
        self.BEST_MODEL_PATH = os.path.join(self.EVALUATED_MODEL_DIR, TRAINED_MODEL_DIR,BEST_MODEL_DIR )
        self.S3_MODEL_NAME = TRAINED_MODEL_NAME
        self.DEVICE = DEVICE
        self.BATCH: int = 1
        self.SHUFFLE: bool = TRAINED_SHUFFLE
        self.NUM_WORKERS = TRAINED_NUM_WORKERS
        self.TRAINED_MODEL_DIR: str = os.path.join(from_root(), ARTIFACTS_DIR, MODEL_TRAINER_ARTIFACTS_DIR,TRAINED_MODEL_DIR)
        self.TRAINED_MODEL_PATH = os.path.join(self.TRAINED_MODEL_DIR, TRAINED_MODEL_NAME)       
        self.S3_MODEL_FOLDER = TRAINED_MODEL_DIR
        self.BUCKET_FOLDER_NAME = BUCKET_FOLDER_NAME
        
        # self.S3_MODEL_KEY_PATH: str = os.path.join(TRAINED_MODEL_DIR,TRAINED_MODEL_NAME)

        self.S3_BUCKET_NAME = BUCKET_NAME