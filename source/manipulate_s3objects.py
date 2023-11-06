import os
import s3fs
import pickle

# Create filesystem object for external S3 endpoint like minio
def create_filesystem_externalS3():
    S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
    fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': S3_ENDPOINT_URL})
    global FS
    FS = fs

# Create filesystem object for AWS
def create_filesystem(region): 
    fs = s3fs.S3FileSystem(client_kwargs={'region_name': region})
    global FS
    FS = fs


def open_text_file(FILE_PATH_S3):
    with FS.open(FILE_PATH_S3, mode="rb") as file_in:
        text_file = file_in.read().decode()
    return text_file

def write_text_file(FILE_PATH_OUT_S3, file_to_save):
    with FS.open(FILE_PATH_OUT_S3, 'w') as file_out:
        file_out.write(file_to_save)

def open_pickle_file(FILE_PATH_S3):
    with FS.open(FILE_PATH_S3, mode="rb") as file_in:
        pickle_file = pickle.loads(file_in.read())
    return pickle_file

def save_model(FILE_PATH_OUT_S3, model_name, model):
    with FS.open(FILE_PATH_OUT_S3 + model_name, 'w') as file_out:
        file_out.save(model)

