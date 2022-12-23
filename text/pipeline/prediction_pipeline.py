import os
import io
import sys
import keras
import pickle
from PIL import Image
from transformers import AutoTokenizer
from text.logger import logging
from text.constants import *
from text.exception import CustomException
from text.configurations.s3_syncer import S3Sync
from text.components.data_transformation import DataTransformation
from text.entity.config_entity import DataTransformationConfig
from text.entity.artifacts_entity import DataIngestionArtifacts
from transformers import TFAutoModelForSeq2SeqLM, DataCollatorForSeq2Seq

class PredictionPipeline:
    def __init__(self):
        self.bucket_name = BUCKET_NAME
        # self.model_name = MODEL_NAME
        self.model_path = os.path.join("artifacts", "PredictModel")
        self.data_transformation = DataTransformation(data_transformation_config= DataTransformationConfig,data_ingestion_artifact=DataIngestionArtifacts)

    def get_model_from_s3(self) -> str:
        """
        Method Name :   predict
        Description :   This method predicts the image.
        Output      :   Predictions
        """
        logging.info("Entered the get_model_from_s3 method of PredictionPipeline class")
        try:
            # Loading the best model from s3 bucket 
            prediction_model_path=os.path.join("PredictModel")
            os.makedirs(prediction_model_path, exist_ok=True)
            s3_sync = S3Sync()

            s3_sync.sync_folder_from_s3(folder=prediction_model_path,bucket_name=BUCKET_NAME,bucket_folder_name=BUCKET_FOLDER_NAME)

            best_model_path = os.path.join(prediction_model_path)

            logging.info("Exited the get_model_from_s3 method of PredictionPipeline class")
            return best_model_path

        except Exception as e:
            raise CustomException(e, sys) from e



    def predict(self,best_model_path,text):

        logging.info("Running the predict function")
        try:
            # best_model_path:str = self.get_model_from_gcloud()
            # load_model=keras.models.load_model(best_model_path)
            # with open('tokenizer.pickle', 'rb') as handle:
            #     load_tokenizer = pickle.load(handle)
            
            # text=self.data_transformation.concat_data_cleaning(text)
            # text = [text]            
            # print(text)
            # seq = load_tokenizer.texts_to_sequences(text)
            # padded = pad_sequences(seq, maxlen=300)
            # print(seq)
            # pred = load_model.predict(padded)
            # pred
            # print("pred", pred)
            # if pred>0.3:

            #     print("hate and abusive")
            #     return "hate and abusive"
            # else:
            #     print("no hate")
            #     return "no hate"
            model_checkpoint = "t5-small"
            model_name = best_model_path
            tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
            model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)
            document = text
            if 't5' in model_name: 
                document = "summarize: " + document
            tokenized = tokenizer([document], return_tensors='np')
            out = model.generate(**tokenized, max_length=128)
            with tokenizer.as_target_tokenizer():
                print(tokenizer.decode(out[0]))
                return tokenizer.decode(out[0])
        except Exception as e:
            raise CustomException(e, sys) from e



    def run_pipeline(self,text):
        logging.info("Entered the run_pipeline method of PredictionPipeline class")
        try:

            best_model_path: str = self.get_model_from_s3() 
            predicted_text = self.predict(best_model_path,text)
            logging.info("Exited the run_pipeline method of PredictionPipeline class")
            return predicted_text
        except Exception as e:
            raise CustomException(e, sys) from e


            
