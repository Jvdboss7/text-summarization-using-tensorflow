from dataclasses import dataclass
# Data ingestion artifacts
@dataclass
class DataIngestionArtifacts:
    train_file_path: str
    test_file_path: str
    valid_file_path: str
    data_dict_file_path: str
    all_dataset_file_path:str

@dataclass
class DataTransformationArtifacts:
    tokenized_datasets: str
