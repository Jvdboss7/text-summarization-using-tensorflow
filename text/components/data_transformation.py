import os
import sys
import datasets
import random
import pandas as pd
from text.constants import *
from text.logger import logging
from datasets import load_from_disk
from transformers import AutoTokenizer
from IPython.display import display, HTML
from text.exception import CustomException
from datasets import load_dataset, load_metric
from text.entity.config_entity import DataTransformationConfig
from text.entity.artifacts_entity import DataIngestionArtifacts,DataTransformationArtifacts


class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig, data_ingestion_artifact: DataIngestionArtifacts):
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifact = data_ingestion_artifact

    def load_data(self):
        try:
            logging.info("Loading the data")
            raw_datasets = load_from_disk(self.data_ingestion_artifact.all_dataset_file_path)
            logging.info(f" the dataset: {raw_datasets}")
            return raw_datasets
        except Exception as e:
            raise CustomException(e, sys) from e


    def show_random_samples(self,dataset, num_examples=5):
        try:
            logging.info("Entered the show_random_samples function")
            assert num_examples <= len(
                dataset
            ), "Can't pick more elements than there are in the dataset."
            picks = []
            for _ in range(num_examples):
                pick = random.randint(0, len(dataset) - 1)
                while pick in picks:
                    pick = random.randint(0, len(dataset) - 1)
                picks.append(pick)

            df = pd.DataFrame(dataset[picks])
            for column, typ in dataset.features.items():
                if isinstance(typ, datasets.ClassLabel):
                    df[column] = df[column].transform(lambda i: typ.names[i])
            display(HTML(df.to_html()))
            logging.info("Exited the show_random_function")
        except Exception as e:
            raise CustomException(e, sys) from e


    def preprocess_function(self,examples):
        try:
            logging.info("Entered the preprocess_function")
            # metric = load_metric("rouge")
            # logging.info(f"the model metric is {metric}")
            # fake_preds = FAKE_PREDS
            # fake_labels = FAKE_LABELS
            # metric.compute(predictions=fake_preds, references=fake_labels)
            # Model checkpoint of pretrained model 
            model_checkpoint = "t5-small"
            # Applying Tokenizer 
            tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
            # if model_checkpoint in ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"]:
            #     prefix = "summarize: "
            # else:
            #     prefix = ""

            inputs = [doc for doc in examples["document"]]
            model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True)

            # Setup the tokenizer for targets
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    examples["summary"], max_length=MAX_TARGET_LENGTH, truncation=True
                )

            model_inputs["labels"] = labels["input_ids"]
            logging.info("Exited the preprocess_function")
            return model_inputs
        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_transformation(self):

        try:
            os.makedirs(self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACTS_DIR, exist_ok=True)
            raw_datasets = self.load_data()
            self.show_random_samples(raw_datasets['train'])
            tokenized_datasets = raw_datasets.map(self.preprocess_function, batched=True)
            tokenized_datasets.save_to_disk(self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACTS_DIR)
            data_transformation_artifact = DataTransformationArtifacts(tokenized_datasets=tokenized_datasets)
            logging.info(f"{data_transformation_artifact}")             
            logging.info("Exited the initiate_data_transfomation function")
            return data_transformation_artifact 

        except Exception as e:
            raise CustomException(e, sys) from e
