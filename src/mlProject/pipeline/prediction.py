import pandas as pd
import joblib
from mlProject.s3_connection import create_s3_client
from io import BytesIO
from mlProject import logger
from dotenv import load_dotenv
import os
load_dotenv()
from mlProject.utils.common import read_yaml
from mlProject.constants import CONFIG_FILE_PATH

config = read_yaml(CONFIG_FILE_PATH)
bucket = config.model_evaluation.bucket_name


class PredictionPipeline:
    def __init__(self, s3_model_path):
        self.s3_model_path = s3_model_path
        self.s3 = create_s3_client()
        self.model = self.load_model()

    def load_model(self):
        try:
            if not self.s3:
                logger.info("S3 client not available.")
                return None

            
            # Load the model directly from S3 without downloading
            model_bytes = self.s3.get_object(Bucket=bucket, Key=self.s3_model_path)['Body'].read()
            model = joblib.load(BytesIO(model_bytes))

            return model
        except Exception as e:
            logger.info(f"Error loading model from S3: {str(e)}")
            return None

    def predict(self, input_data):
        if self.model is None:
            logger.info("Model not loaded.")
            return None

        # Perform prediction
        predictions = self.model.predict(input_data)

        return predictions

# if __name__ == "__main__":
#     # Example usage
#     s3_model_path = "model/model_20231224122240/model.pkl"  # Update with the actual S3 model path
#     prediction_pipeline = PredictionPipeline(s3_model_path)

#     # Input data for prediction
#     input_data = [[7.8, 0.64, 0.1, 6.0, 0.115, 5.0, 11.0, 0.9984, 3.37, 0.69, 10.1]]

#     # Perform prediction
#     predictions = prediction_pipeline.predict(input_data)

#     # Display predictions
#     print("Predictions:")
#     print(predictions)


# import mlflow
# import mlflow.sklearn
# import joblib

# class PredictionPipeline:
#     def __init__(self, s3_bucket_name, mlflow_run_id):
#         self.s3_bucket_name = s3_bucket_name
#         self.mlflow_run_id = mlflow_run_id
#         self.loaded_model = None

#     def load_model(self):
#         try:
#             # Construct the model path
#             model_path = f's3://{self.s3_bucket_name}/mlflow/{self.mlflow_run_id}/artifacts/model'
            
#             # Load the model using MLflow's API
#             self.loaded_model = mlflow.sklearn.load_model(model_path)
#             print("Model loaded successfully.")
#         except Exception as e:
#             print(f"Error loading model: {e}")

#     def predict(self, input_data):
#         if self.loaded_model is not None:
#             try:
#                 # Make predictions using the loaded model
#                 predictions = self.loaded_model.predict(input_data)
#                 return predictions
#             except Exception as e:
#                 print(f"Error making predictions: {e}")
#         else:
#             print("Model is not loaded. Call load_model() first.")

# # Example usage:
# if __name__ == "__main__":
#     # Replace 'winequalitypre' and 'your-mlflow-run-id' with actual values
#     s3_bucket_name = 'winequalitypre'
#     mlflow_run_id = 'your-mlflow-run-id'

#     # Create an instance of PredictionPipeline
#     prediction_pipeline = PredictionPipeline(s3_bucket_name, mlflow_run_id)

#     # Load the model
#     prediction_pipeline.load_model()

#     # Example: Make predictions
#     input_data = [8.0,0.59,0.05,2.0,0.089,12.0,32.0,0.99735,3.36,0.61,10.0]  # Replace with your input data
#     predictions = prediction_pipeline.predict(input_data)
#     print("Predictions:", predictions)



# import mlflow
# import joblib
# s3_bucket_name = 'winequalitypre'
# s3_object_key = f"{mlflow.active_run().info.run_id}/artifacts/model/model.pkl"
# model_path = f's3://{s3_bucket_name}/{s3_object_key}'
# model = joblib.load(model_path)



# import joblib 
# import numpy as np
# import pandas as pd
# from pathlib import Path


# class PredictionPipeline:
#     def __init__(self):
#         self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))

    
#     def predict(self, data):
#         prediction = self.model.predict(data)

#         return prediction
