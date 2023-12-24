import os
import pandas as pd
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from urllib.parse import urlparse
from dotenv import load_dotenv
from mlProject import logger
from datetime import datetime
from mlProject.s3_connection import upload_to_s3  # Adjust the import as needed
from mlProject.entity.config_entity import ModelEvaluationConfig
from mlProject.utils.common import save_json
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlProject.utils.common import save_json
from mlProject.entity.config_entity import ModelEvaluationConfig
from pathlib import Path


load_dotenv()

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def log_into_mlflow(self):
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            predicted_qualities = model.predict(test_x)

            (rmse, mae, r2) = self.eval_metrics(test_y, predicted_qualities)

            # Saving metrics as local
            scores = {"rmse": rmse, "mae": mae, "r2": r2}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            mlflow.log_params(self.config.all_params)

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)

            # Model registry does not work with file store
            if tracking_url_type_store != "file":
                # Register the model
                mlflow.sklearn.log_model(model, "model", registered_model_name="ElasticnetModel")

                # Use MLflow URI to store the model version
                mlflow_model_uri = mlflow.get_artifact_uri("model")
                # s3_model_path = f"{mlflow_model_uri}/model.pkl"
                
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                s3_model_path = f"model/model_{timestamp}/model.pkl"
                print(f"Debug: tracking_url_type_store={tracking_url_type_store}, s3_model_path={s3_model_path}")

                # Upload the model to S3
                upload_to_s3(self.config.model_path, s3_model_path)
            else:
                mlflow.sklearn.log_model(model, "model")





# import os
# import pandas as pd
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from mlProject.utils.common import save_json
# from mlProject.entity.config_entity import ModelEvaluationConfig
# from pathlib import Path
# from urllib.parse import urlparse
# import mlflow
# import mlflow.sklearn
# import numpy as np
# import joblib
# from dotenv import load_dotenv
# from botocore.exceptions import NoCredentialsError
# from mlProject import logger
# from datetime import datetime
# from mlProject.s3_connection import create_s3_client  # Import the S3 connection function

# load_dotenv()

# class ModelEvaluation:
#     def __init__(self, config: ModelEvaluationConfig):
#         self.config = config
#         self.s3 = create_s3_client()  # Use the S3 connection

#     def eval_metrics(self, actual, pred):
#         rmse = np.sqrt(mean_squared_error(actual, pred))
#         mae = mean_absolute_error(actual, pred)
#         r2 = r2_score(actual, pred)
#         return rmse, mae, r2

#     def upload_to_s3(self, local_path, s3_path):
#         try:
#             if not self.s3:
#                 logger.info("S3 client not available.")
#                 return

#             # Upload the file
#             self.s3.upload_file(local_path, os.getenv("S3_BUCKET_NAME"), s3_path)

#             logger.info(f"Model uploaded to {s3_path}")
#         except FileNotFoundError:
#             logger.info(f"The file {local_path} was not found.")
#         except NoCredentialsError:
#             logger.info("Credentials not available.")

#     def log_into_mlflow(self):
#         test_data = pd.read_csv(self.config.test_data_path)
#         model = joblib.load(self.config.model_path)

#         test_x = test_data.drop([self.config.target_column], axis=1)
#         test_y = test_data[[self.config.target_column]]

#         mlflow.set_registry_uri(self.config.mlflow_uri)
#         tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

#         with mlflow.start_run():
#             predicted_qualities = model.predict(test_x)

#             (rmse, mae, r2) = self.eval_metrics(test_y, predicted_qualities)

#             # Saving metrics as local
#             scores = {"rmse": rmse, "mae": mae, "r2": r2}
#             save_json(path=Path(self.config.metric_file_name), data=scores)

#             mlflow.log_params(self.config.all_params)

#             mlflow.log_metric("rmse", rmse)
#             mlflow.log_metric("r2", r2)
#             mlflow.log_metric("mae", mae)

#             # Model registry does not work with file store
#             if tracking_url_type_store != "file":
#                 # Register the model
#                 mlflow.sklearn.log_model(model, "model", registered_model_name="ElasticnetModel")

#                 timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

#                 # Upload the model to S3
#                 s3_model_path = f"model/model_{timestamp}/model.pkl"
#                 print(f"Debug: tracking_url_type_store={tracking_url_type_store}, s3_model_path={s3_model_path}")
#                 self.upload_to_s3(self.config.model_path, s3_model_path)
#             else:
#                 mlflow.sklearn.log_model(model, "model")
