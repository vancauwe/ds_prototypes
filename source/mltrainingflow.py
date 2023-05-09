#from manipulate_s3objects import create_filesystem, open_text_file, open_pickle_file
from trainer import define_model, train_model
from preprocessing import preprocessing
import mlflow
import mlflow.keras
from metaflow import FlowSpec, step
import s3fs
import pickle 
import os 

class MLTrainingFlow(FlowSpec):

    def open_text_file(FILE_PATH_S3, FS):
        with FS.open(FILE_PATH_S3, mode="rb") as file_in:
            text_file = file_in.read().decode()
        return text_file

    def write_text_file(FILE_PATH_OUT_S3, FS, file_to_save):
        with FS.open(FILE_PATH_OUT_S3, 'w') as file_out:
            file_out.write(file_to_save)

    def open_pickle_file(FILE_PATH_S3, FS):
        with FS.open(FILE_PATH_S3, mode="rb") as file_in:
            pickle_file = pickle.loads(file_in.read())
        return pickle_file

    @step
    def start(self):
        # Create filesystem for S3 storage
        self.next(self.create_filesystem)

    @step
    def create_filesystem(self):
        S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
        print(S3_ENDPOINT_URL)
        fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': S3_ENDPOINT_URL})
        self.FS = fs
        self.next(self.open_files)

    @step
    def open_files(self):
    
        BUCKET = "lvancauwe/image_caption_generator"

        # Defining file paths and opening data 
        FILE_KEY_S3 = "Flickr8k.token.txt"
        self.FILE_PATH_S3 = BUCKET + "/" + FILE_KEY_S3
        self.text_file_captions = self.open_text_file(self.FILE_PATH_S3, self.FS)

        FILE_KEY_S3 = "Flickr_8k.trainImages.txt"
        self.FILE_PATH_S3 = BUCKET + "/" + FILE_KEY_S3
        self.file_train_images = self.open_text_file(self.FILE_PATH_S3, self.FS)

        FILE_KEY_S3 = "features.p"
        self.FILE_PATH_S3 = BUCKET + "/" + FILE_KEY_S3
        self.all_features = self.open_pickle_file(self.FILE_PATH_S3, self.FS)
        
        self.next(self.preprocessing)

    

    


    @step
    def preprocessing(self):
        print('Preprocessing')
        self.all_descriptions, self.train_descriptions, self.train_features, self.tokenizer, self.vocab_size, self.max_length = preprocessing(self.text_file_captions, self.file_train_images, self.all_features)
        self.next(self.create_model)

    @step
    def create_model(self):
        print('Model Creation')
        self.model = define_model(self.vocab_size, self.max_length)
        self.next(self.train)

    @step
    def train(self):
        print('Model Training')
        mlflow.keras.autolog()
        self.experiment_id = mlflow.create_experiment("experiment_full_flow")
        # making a directory models to save our models
        BUCKET_OUT = "lvancauwe"
        FILE_KEY_OUT_S3 = "image_caption_generator/models/"
        FILE_PATH_OUT_S3 = BUCKET_OUT + "/*" + FILE_KEY_OUT_S3
        train_model(self.model, self.train_descriptions, self.train_features, self.tokenizer, self.max_length, self.vocab_size, FILE_PATH_OUT_S3)
        self.next(self.end)

    @step
    def end(self):
        print('Finished Training')


if __name__ == '__main__':
    MLTrainingFlow()






