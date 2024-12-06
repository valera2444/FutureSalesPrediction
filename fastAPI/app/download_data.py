
import os, boto3
import argparse

class Args:
    def __init__(self, d: dict):
        for el in d.keys():
            setattr(self, el, d[el])

def download_s3_folder(s3c, bucket_name, s3_folder, local_dir=None):
    """
    Download the contents of a folder directory to local_dir (creates if not exist)
    Args:
        bucket_name: the name of the s3 bucket
        s3_folder: the folder path in the s3 bucket
        local_dir: a relative or absolute directory path in the local file system
    """
    bucket = s3c.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=s3_folder):
        target = obj.key if local_dir is None \
            else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == '/':
            continue
        bucket.download_file(obj.key, target)
        #print(obj.key, target)

def download_from_minio(args):

    minio_user=os.environ.get("MINIO_ACCESS_KEY")
    minio_password=os.environ.get("MINIO_SECRET_ACCESS_KEY")
    bucket_name = os.environ.get("BUCKET_NAME")
    
    ACCESS_KEY = minio_user
    SECRET_KEY =  minio_password
    host = 'http://minio:9000'
    bucket_name = bucket_name

    s3c = boto3.resource('s3', 
                    aws_access_key_id=ACCESS_KEY,
                    aws_secret_access_key=SECRET_KEY,
                    endpoint_url=host) 
    
    download_s3_folder(s3c,bucket_name,args.path_data_cleaned, args.path_data_cleaned) # this must be used only once on server part 

    s3c = boto3.client('s3', 
                    aws_access_key_id=ACCESS_KEY,
                    aws_secret_access_key=SECRET_KEY,
                    endpoint_url=host)
    
    s3c.download_file(bucket_name, f'{args.path_for_merged}/merged.csv', f'{args.path_for_merged}/merged.csv') # ASSUMES THAT path args.path_for_merged exists +  this must be used only once on server part 
    
    s3c.download_file(bucket_name, f'{args.run_name}/lgbm.pkl', f'lgbm.pkl')

