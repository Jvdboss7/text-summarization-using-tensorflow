# text-summarization-using-tensorflow

## Problem statement
The problem introduces a Text summarization tast, which comes under NLP. In this project we take paragraphs as inputs and we get output as the summary of the paragraph.

## Solution proposed
The solution proposed for the above problem is that we have used NLP to solve the above problem. We have used the Tensorflow framework to solve the above problem also we used t5-small model from hugging face library. Then we created an API that takes in paragraph and gives out the summary. Then we dockerized the application and deployed the model on AWS cloud.

## Dataset used
In the project the dataset contains train, test and validation data. The train data is of size (204045,3), test data is of size (11334,3) and validation data is of size (11332,3).  

## Tech stack used
1. Python 3.8
2. FastAPI
3. Deep learning
4. Natural language preocessing - t5-small model
5. Docker

## Infrastructure required
1. AWS S3
2. AWS EC2 instance
3. AWS ECR
4. GitHub Actions

## How to run

Step 1. Cloning the repository.
```
git clone https://github.com/Deep-Learning-01/image-captioning.git
```
Step 2. Create a conda environment.
```
conda create -p env python=3.8 -y
```
```
conda activate env/
```
Step 3. Install the requirements
```
pip install -r requirements.txt
```
Step 4. Export the environment variables
```bash
export AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID>
export AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY>
export AWS_DEFAULT_REGION=<AWS_DEFAULT_REGION>
```
Step 5. Run the application server
```
python app.py
```
Step 6. Train application
```
http://localhost:8080/train
```
Step 7. Prediction application
```
http://localhost:8080/predict
```

## Run locally
1. Check if the Dockerfile is available in the project directory.
2. Build the Docker image
```
docker build --build-arg AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID> --build-arg AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY> --build-arg AWS_DEFAULT_REGION=<AWS_DEFAULT_REGION> . 
```
3. Run the docker image
```
docker run -d -p 8080:8080 <IMAGE_NAME>
```

## `text` is the main package folder which contains -

**Components** : Contains all components of this Project
- DataIngestion
- DataTransformation
- ModelTrainer
- ModelEvaluation
- ModelPusher

**Custom Logger and Exceptions** are used in the Project for better debugging purposes.

## Conclusion
- We have created a API which takes in Paragraphs as input and give the output as summary of the paragraphs.