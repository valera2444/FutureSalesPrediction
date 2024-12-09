
from google.cloud import storage
import os
import glob
from google.cloud import storage

from pathlib import Path



def upload_file(filename, bucket_name):
    """Uploads a file to the bucket."""
  
    storage_client = storage.Client.from_service_account_json('service_account_token.json')
  
    bucket = storage_client.get_bucket(bucket_name)

    blob = bucket.blob(filename)

    blob.upload_from_filename(filename)

    print('File {} uploaded to {}.'.format(
        filename,
        filename))


def upload_folder(path, bucket_name):

    storage_client = storage.Client.from_service_account_json('service_account_token.json')
    bucket = storage_client.get_bucket(bucket_name)

    for local_file in glob.glob(path + '/**'):
        if not os.path.isfile(local_file):
           upload_folder(local_file, bucket_name, path + "/" + os.path.basename(local_file))
        else:
           remote_path = os.path.join(path, local_file[1 + len(path):])
           blob = bucket.blob(remote_path)
           blob.upload_from_filename(local_file)




def download_folder(bucket_name,path):
    storage_client = storage.Client.from_service_account_json('service_account_token.json')
    bucket = storage_client.get_bucket(bucket_name)
   
    blobs = bucket.list_blobs(prefix=path)  # Get list of files
    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        file_split = blob.name.split("/")
        directory = "/".join(file_split[0:-1])
        Path(directory).mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(blob.name) 


def download_file(bucket_name, destination_file_name):
    storage_client = storage.Client.from_service_account_json('service_account_token.json')
    bucket = storage_client.get_bucket(bucket_name)
    
    blob = bucket.blob(destination_file_name)
    # Download the file to a destination
    blob.download_to_filename(destination_file_name)



