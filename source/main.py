from manipulate_s3objects import create_filesystem, open_text_file, open_pickle_file
from trainer import define_model, train_model
from preprocessing import preprocessing
import mlflow
import mlflow.keras

# Create filesystem for S3 storage

create_filesystem()

BUCKET = "lvancauwe/image_caption_generator"

# Defining file paths and opening data 

FILE_KEY_S3 = "Flickr8k.token.txt"
FILE_PATH_S3 = BUCKET + "/" + FILE_KEY_S3
text_file_captions = open_text_file(FILE_PATH_S3)

FILE_KEY_S3 = "Flickr_8k.trainImages.txt"
FILE_PATH_S3 = BUCKET + "/" + FILE_KEY_S3
file_train_images = open_text_file(FILE_PATH_S3)

FILE_KEY_S3 = "features.p"
FILE_PATH_S3 = BUCKET + "/" + FILE_KEY_S3
all_features = open_pickle_file(FILE_PATH_S3)

# PREPROCESSING 

all_descriptions, train_descriptions, train_features, tokenizer, vocab_size, max_length = preprocessing(text_file_captions, file_train_images, all_features)


# FILE_KEY_OUT_S3 = "descriptions.txt"
# FILE_PATH_OUT_S3 = BUCKET + "/" + FILE_KEY_OUT_S3
# write_text_file(FILE_PATH_OUT_S3, all_descriptions)

# MODEL TRAINING

model = define_model(vocab_size, max_length)

mlflow.keras.autolog()

# making a directory models to save our models
BUCKET_OUT = "lvancauwe"
FILE_KEY_OUT_S3 = "image_caption_generator/models/"
FILE_PATH_OUT_S3 = BUCKET_OUT + "/*" + FILE_KEY_OUT_S3

train_model(model, train_descriptions, train_features, tokenizer, max_length, vocab_size, FILE_PATH_OUT_S3)