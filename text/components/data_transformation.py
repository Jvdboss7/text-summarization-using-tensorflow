import os
import sys
from text.logger import logging
from text.exception import CustomException
from datasets import load_from_disk
from text.constants import *
from text.entity.config_entity import DataTransformationConfig
from text.entity.artifacts_entity import DataIngestionArtifacts,DataTransformationArtifacts


class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig, data_ingestion_artifact: DataIngestionArtifacts):
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifact = data_ingestion_artifact

    def load_data(self):

        try:
            raw_datasets = load_from_disk(self.data_ingestion_artifact.all_dataset_file_path)
            print(raw_datasets)
            return raw_datasets
        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_transformation(self):

        try:
            read_data = self.load_data()
            print(read_data)

            data_transformation_artifact = DataTransformationArtifacts(raw_datasets=self.data_transformation_config.RAW_DATASET)
            return data_transformation_artifact 

        except Exception as e:
            raise CustomException(e, sys) from e
