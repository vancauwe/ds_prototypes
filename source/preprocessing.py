from manipulate_s3objects import open_text_file, write_text_file

from preprocessing_captions import preprocessing_captions
from preprocessing_features import preprocessing_features
from tokenizer import create_tokenizer

# CAPTIONS
FILE_KEY_S3 = "image_caption_generator/Flickr8k.token.txt"
FILE_PATH_S3 = BUCKET + "/" + FILE_KEY_S3
text_file = open_text_file(FILE_PATH_S3)

descriptions, clean_descriptions, vocabulary, all_descriptions = preprocessing_captions(text_file)

FILE_KEY_OUT_S3 = "image_caption_generator/descriptions.txt"
FILE_PATH_OUT_S3 = BUCKET_OUT + "/" + FILE_KEY_OUT_S3
write_text_file(FILE_PATH_OUT_S3, all_descriptions)

# IMAGES & FEATURES 
FILE_KEY_S3 = "image_caption_generator/Flickr_8k.trainImages.txt"
FILE_PATH_S3 = BUCKET + "/" + FILE_KEY_S3
file_train_images = open_text_file(FILE_PATH_S3)

FILE_KEY_S3 = "image_caption_generator/features.p"
FILE_PATH_S3 = BUCKET + "/" + FILE_KEY_S3
all_features = open_pickle_file(FILE_PATH_S3)

train_imgs, train_descriptions, train_features = preprocessing_features(file_train_images, all_descriptions)

# TOKENIZER
# give each word an index, and store that into tokenizer.p pickle file
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
