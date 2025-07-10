import os
from dotenv import load_dotenv

load_dotenv('.env.dev')

class Config:
    # Kaggle API
    KAGGLE_USERNAME = os.getenv('KAGGLE_USERNAME')
    KAGGLE_KEY = os.getenv('KAGGLE_KEY')
    
    # Hugging Face
    HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
    
    # MLflow
    MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    MLFLOW_EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', 'mental-health-chatbot')
    
    # Model
    MODEL_NAME = os.getenv('MODEL_NAME', 'microsoft/Phi-3-mini-4k-instruct')
    MAX_LENGTH = int(os.getenv('MAX_LENGTH', '512'))
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', '8'))
    
    # Streaming
    STREAMING_BATCH_SIZE = int(os.getenv('STREAMING_BATCH_SIZE', '1000'))
    PREPROCESSING_WORKERS = int(os.getenv('PREPROCESSING_WORKERS', '4'))