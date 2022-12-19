import os
import sys
import pandas as pd
from text.constants import *
from text.logger import logging
from datasets import load_from_disk
from transformers import AutoTokenizer
from transformers import AdamWeightDecay
from text.exception import CustomException
from text.entity.config_entity import ModelTrainerConfig
from transformers import TFAutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from text.entity.artifacts_entity import DataTransformationArtifacts, ModelTrainerArtifacts
class ModelTrainer:
    def __init__(self,data_transformation_artifacts: DataTransformationArtifacts,
                    model_trainer_config: ModelTrainerConfig):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: Configuration for data transformation
        """

        self.data_transformation_artifacts = data_transformation_artifacts
        self.model_trainer_config = model_trainer_config

    def preparing_train_data(self):
        try:
            # tokenized_datasets = load_from_disk(os.path.join(os.getcwd(),"tokenized_data"))
            # tokenized_datasets = load_from_disk(os.path.join(os.getcwd(),"artifacts/12_19_2022_12_02_25/DataTransformationArtifacts/" ))
            print(self.data_transformation_artifacts.tokenized_datasets)
            tokenized_datasets = load_from_disk(self.data_transformation_artifacts.path_tokenized_data)
            small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(6000))
            small_validation_dataset = tokenized_datasets["validation"].shuffle(seed=42).select(range(1000))
            small_test_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
            
            # Model checkpoint of pretrained model 
            model_checkpoint = "t5-small"

            model = TFAutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
            model_name = model_checkpoint.split("/")[-1]
            push_to_hub_model_id = f"{model_name}-finetuned-xsum"
            tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
            data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="tf")

            generation_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="tf", pad_to_multiple_of=128)
            train_dataset = model.prepare_tf_dataset(
                small_train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                collate_fn=data_collator,
            )

            validation_dataset = model.prepare_tf_dataset(
                small_validation_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                collate_fn=data_collator,
            )

            test_dataset = model.prepare_tf_dataset(
                small_test_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                collate_fn=data_collator,
            )

            optimizer = AdamWeightDecay(learning_rate=LEARNING_RATE, weight_decay_rate=WEIGHT_DECAY)
            model.compile(optimizer=optimizer)

            return model, train_dataset, validation_dataset, test_dataset
            
        except Exception as e:
            raise CustomException(e, sys) from e

    def start_model_training(self,model,train_dataset,validation_dataset):
        try:
            history = model.fit(train_dataset, validation_data=validation_dataset, epochs=EPOCHS)
            # logging.info(f"the loss of the trained model is {history.history}")

            return history
        except Exception as e:
            raise CustomException(e, sys) from e

   

    def initiate_model_trainer(self,) -> ModelTrainerArtifacts:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates a model trainer steps
        
        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """

        try:
            
            model,train_dataset,validation_dataset,test_dataset = self.preparing_train_data()

            history= self.start_model_training(model=model,train_dataset=train_dataset,validation_dataset=validation_dataset)
            logging.info(f"the loss of the trained model is {history.history}")

            os.makedirs(self.model_trainer_config.TRAINED_MODEL_DIR, exist_ok=True)
            # torch.save(model, self.model_trainer_config.TRAINED_MODEL_PATH)
            # saving the pretrained model
            model.save_pretrained(self.model_trainer_config.TRAINED_MODEL_PATH)
            logging.info(f"Saved the trained model")

            model_trainer_artifacts = ModelTrainerArtifacts(
                trained_model_path=self.model_trainer_config.TRAINED_MODEL_PATH
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifacts}")

            return model_trainer_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e



