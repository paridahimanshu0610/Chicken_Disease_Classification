import os
import shutil
import gdown
import urllib.request as request
import zipfile
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import DataIngestionConfig
from pathlib import Path
from sklearn.model_selection import train_test_split


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config


    
    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url = self.config.source_URL,
                filename = self.config.local_data_file
            )
            logger.info(f"{filename} downloaded! with following info: \n{headers}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")  

    def download_file_from_gdrive(self):
        if not os.path.exists(self.config.local_data_file):
            filename = gdown.download(
                url = self.config.source_URL,
                output = self.config.local_data_file,
                quiet=False,
                fuzzy=True
            )
            logger.info(f"{filename} downloaded!")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}") 
    
    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)


    def extract_and_split_data(self):
        # Define paths for temporary extraction and final destination
        temp_extr_dir = os.path.join(self.config.unzip_dir, 'temp_dir')
        train_data_path = self.config.train_data_dir
        test_data_path = self.config.test_data_dir

        # Create temporary extraction folder
        os.makedirs(temp_extr_dir, exist_ok=True)

        # Extract the zip file to the temporary extraction folder
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(temp_extr_dir)

        # Get the list of files in the temporary extraction folder (including subdirectories)
        file_list = []
        for root, _, files in os.walk(temp_extr_dir):
            for file in files:
                file_list.append(os.path.join(root, file))

        # Extract class information from file paths
        classes = [os.path.basename(os.path.dirname(file)) for file in file_list]
        
        # Split files into train and test sets
        train_files, test_files = train_test_split(file_list, test_size=0.2, random_state=0, stratify=classes)
                                            

        # Create train and test data folders
        os.makedirs(train_data_path, exist_ok=True)
        os.makedirs(test_data_path, exist_ok=True)

        # Extract files into train and test folders
        for files, dest_path in zip([train_files, test_files], [train_data_path, test_data_path]):
            for file in files:
                class_folder = os.path.basename(os.path.dirname(file))
                dest_folder = os.path.join(dest_path, class_folder)

                # Create the destination folder if it doesn't exist
                os.makedirs(dest_folder, exist_ok=True)

                dest_file_path = os.path.join(dest_folder, os.path.basename(file))
                shutil.copy2(file, dest_file_path)  # Using shutil.copy2 to preserve file metadata

        # Delete the temporary extraction folder
        shutil.rmtree(temp_extr_dir)