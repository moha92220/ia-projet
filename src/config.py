import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_NAME = os.getenv("PROJECT_NAME", "tp-ml-devops")
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))

DATA_RAW_DIR = os.getenv("DATA_RAW_DIR", "data/raw")
DATA_PROCESSED_DIR = os.getenv("DATA_PROCESSED_DIR", "data/processed")
MODELS_DIR = os.getenv("MODELS_DIR", "models")
REPORTS_DIR = os.getenv("REPORTS_DIR", "reports")
