from dataclasses import dataclass

# Data ingestion artifacts
@dataclass
class DataIngestionArtifacts:
    train_file_path: str
    test_file_path: str
    valid_file_path: str
    data_dict_file_path: str
    all_dataset_file_path:str

# Data Transformation Artifacts
@dataclass
class DataTransformationArtifacts:
    tokenized_datasets: str
    path_tokenized_data: str

# Model Trainer artifacts
@dataclass
class ModelTrainerArtifacts:
    trained_model_path: str
    
@dataclass
class ModelEvaluationArtifacts:
    is_model_accepted: bool
    all_losses: str