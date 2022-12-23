import os
import sys
import math
import pandas as pd
import numpy as np
from text.constants import *
from text.logger import logging
from transformers import AdamWeightDecay
from datasets import load_from_disk
from text.exception import CustomException
from text.components.model_trainer import ModelTrainer
from text.entity.config_entity import ModelEvaluationConfig, ModelTrainerConfig
from text.entity.artifacts_entity import ModelTrainerArtifacts, DataTransformationArtifacts, ModelEvaluationArtifacts
from text.configurations.s3_syncer import S3Sync
from transformers import TFAutoModelForSeq2SeqLM, DataCollatorForSeq2Seq


class ModelEvaluation:

    def __init__(self, model_evaluation_config:ModelEvaluationConfig,
                data_transformation_artifacts:DataTransformationArtifacts,
                model_trainer_artifacts:ModelTrainerArtifacts,
                ):

        self.model_evaluation_config = model_evaluation_config
        self.data_transformation_artifacts = data_transformation_artifacts
        self.model_trainer_artifacts = model_trainer_artifacts
        self.s3 = S3Sync()
        self.bucket_name = BUCKET_NAME
        self.model_trainer =ModelTrainer(data_transformation_artifacts=DataTransformationArtifacts, model_trainer_config=ModelTrainerConfig())

    # @staticmethod
    # def collate_fn(batch):
    #     """
    #     This is our collating function for the train dataloader,
    #     it allows us to create batches of data that can be easily pass into the model
    #     """
    #     try:
    #         return tuple(zip(*batch))
    #     except Exception as e:
    #         raise CustomException(e, sys) from e

    def get_model_from_s3(self):
        """
        Method Name :   predict
        Description :   This method predicts the image.

        Output      :   Predictions
        """
        logging.info("Entered the get_model_from_s3 method of PredictionPipeline class")
        try:
            logging.info(f"Checking the s3_key path{self.model_evaluation_config.TRAINED_MODEL_PATH}")
            print(f"s3_key_path:{self.model_evaluation_config.TRAINED_MODEL_PATH}")
            best_model = self.s3.s3_key_path_available(bucket_name=self.model_evaluation_config.S3_BUCKET_NAME,s3_key="ModelTrainerArtifacts/trained_model/")

            if best_model:
                self.s3.sync_folder_from_s3(folder=self.model_evaluation_config.EVALUATED_MODEL_DIR,bucket_name=self.model_evaluation_config.S3_BUCKET_NAME,bucket_folder_name=self.model_evaluation_config.BUCKET_FOLDER_NAME)
            logging.info("Exited the get_model_from_s3 method of PredictionPipeline class")
           # best_model_path = os.path.join(self.model_evaluation_config.EVALUATED_MODEL_DIR)
            best_model_path = TFAutoModelForSeq2SeqLM.from_pretrained(self.model_evaluation_config.EVALUATED_MODEL_DIR)

            return best_model_path

        except Exception as e:
            raise CustomException(e, sys) from e



    def initiate_model_evaluation(self) -> ModelEvaluationArtifacts:
        """
                Method Name :   initiate_model_evaluation
                Description :   This function is used to initiate all steps of the model evaluation

                Output      :   Returns model evaluation artifact
                On Failure  :   Write an exception log and then raise an exception
        """

        try:
            
            #tokenized_datasets = load_from_disk(self.data_transformation_artifacts.path_tokenized_data)            
            # model = self.model_trainer_artifacts.trained_model_path
            model,train_dataset,validation_dataset,test_dataset = self.model_trainer.preparing_train_data()
            loss = model.evaluate(test_dataset)
            print(loss)
            

            # trained_model = trained_model.to(DEVICE)

            # all_losses_dict, all_losses = self.evaluate(trained_model, test_loader, device=DEVICE)

            os.makedirs(self.model_evaluation_config.EVALUATED_MODEL_DIR, exist_ok=True)
            
            #loss.to_csv(self.model_evaluation_config.EVALUATED_LOSS_CSV_PATH, index=False)

            s3_model = self.get_model_from_s3()
            optimizer = AdamWeightDecay(learning_rate=LEARNING_RATE, weight_decay_rate=WEIGHT_DECAY)
            s3_model.compile(optimizer=optimizer)
            logging.info(f"{s3_model}")

            is_model_accepted = False
            s3_loss = None 
            print(f"--------------------------{s3_model}--------------------------------")
            #print(f"{os.path.isfile(s3_model)}")
            if s3_model is False: 
                is_model_accepted = True
                print("s3 model is false and model accepted is true")
                s3_loss = None

            else:
                print("Entered inside the else condition")

                s3_model =s3_model.evaluate(test_dataset)

                print("Model loaded from s3")
                s3_loss = model.evaluate(test_dataset)

                if s3_loss > loss:
                    print(f"printing the loss inside the if condition{s3_loss} and {loss}")
                    # 0.03 > 0.02
                    is_model_accepted = True
                    print("f{is_model_accepted}")
            model_evaluation_artifact = ModelEvaluationArtifacts(
                        is_model_accepted=is_model_accepted)
            print(f"{model_evaluation_artifact}")

            logging.info("Exited the initiate_model_evaluation method of Model Evaluation class")
            return model_evaluation_artifact

        except Exception as e:
            raise CustomException(e, sys) from e
