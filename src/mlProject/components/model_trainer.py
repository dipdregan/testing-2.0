import pandas as pd
import os
from mlProject import logger
from sklearn.linear_model import ElasticNet
from mlProject.entity.config_entity import ModelTrainerConfig
import joblib
from dotenv import load_dotenv
load_dotenv()

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]]
        test_y = test_data[[self.config.target_column]]

        # Check if the model file already exists
        model_path = os.path.join(self.config.root_dir, self.config.model_name)
        if os.path.exists(model_path):
            # If the model file exists, remove it
            os.remove(model_path)
            logger.info(f"Removed existing model file: {model_path}")

        lr = ElasticNet(alpha=self.config.alpha, l1_ratio=self.config.l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        # Save the new model
        joblib.dump(lr, model_path)
        logger.info(f"New model saved at: {model_path}")
