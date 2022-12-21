from text.pipeline.train_pipeline import TrainPipeline

if __name__ == "__main__":
    TrainPipeline().run_pipeline()


# from text.components.data_ingestion import DataIngestion
# from text.components.data_transformation import DataTransformation
# from text.components.model_trainer import ModelTrainer
# from text.components.model_evaluation import ModelEvaluation
# from text.entity.config_entity import DataIngestionConfig, DataTransformationConfig, ModelTrainerConfig, ModelEvaluationConfig

# data_ingestion = DataIngestion(data_ingestion_config=DataIngestionConfig())

# data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

# data_transformation = DataTransformation(data_transformation_config=DataTransformationConfig(), data_ingestion_artifact=data_ingestion_artifact)

# data_transformation_artifact = data_transformation.initiate_data_transformation()

# model_trainer = ModelTrainer(data_transformation_artifacts=data_transformation_artifact, model_trainer_config=ModelTrainerConfig())

# model_trainer_artifact = model_trainer.initiate_model_trainer()

# model_evaluation = ModelEvaluation(model_evaluation_config=ModelEvaluationConfig(), data_transformation_artifacts=data_transformation_artifact, model_trainer_artifacts=model_trainer_artifact)

# model_evaluation.initiate_model_evaluation()



#artifacts/12_19_2022_12_02_25/DataTransformationArtifacts/" ))
# from text.entity.config_entity import DataTransformationConfig
# config = DataTransformationConfig()
# print(config.DATA_TRANSFORMATION_ARTIFACTS_DIR) 
# import os
# from text.constants import *
# print(os.path.join(os.getcwd(),ARTIFACTS_DIR,DATA_TRANSFORMATION_ARTIFACTS_DIR + '/'))