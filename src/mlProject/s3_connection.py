import boto3
import os
from dotenv import load_dotenv
from botocore.exceptions import NoCredentialsError
from mlProject import logger  # Replace with the correct import for your logger

load_dotenv()

def create_s3_client():
    try:
        # Create an S3 client
        s3 = boto3.client(
            's3',
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION")
        )
        return s3
    except Exception as e:
        logger.error(f"Error creating S3 client: {str(e)}")
        return None

def upload_to_s3(local_path, s3_path):
    try:
        s3 = create_s3_client()
        if not s3:
            logger.error("S3 client not available.")
            return
        # Upload the file
        s3.upload_file(local_path,"balti901", s3_path)
        logger.info(f"File uploaded to S3 at {s3_path}")
    except FileNotFoundError:
        logger.error(f"The file {local_path} was not found.")
    except NoCredentialsError:
        logger.error("Credentials not available.")

def download_from_s3(s3_path, local_path):
    try:
        s3 = create_s3_client()
        if not s3:
            logger.error("S3 client not available.")
            return
        # Download the file
        s3.download_file("balti901", s3_path, local_path)
        logger.info(f"File downloaded from S3 to {local_path}")
    except NoCredentialsError:
        logger.error("Credentials not available.")
    except Exception as e:
        logger.error(f"Error downloading file from S3: {str(e)}")

if __name__ == "__main__":
    # Specify local and S3 paths
    local_path = r"F:\End_To_End_project\MlFLOW_S3\mlflow2.0\mlflow_testing_with_aws\artifacts\model_trainer\model.pkl"
    s3_path = "testing/model.pkl"

    # Upload file to S3
    upload_to_s3(local_path, s3_path)

    # Download file from S3
    download_from_s3(s3_path, r'../artifacts')

