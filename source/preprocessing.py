from preprocessing_captions import preprocessing_captions
from preprocessing_features import preprocessing_features
from tokenizer import create_tokenizer

def preprocessing(text_file_captions, file_train_images, all_features): 
    # CAPTIONS
    descriptions, clean_descriptions, vocabulary, all_descriptions = preprocessing_captions(text_file_captions)

    # IMAGES & FEATURES 
    train_imgs, train_descriptions, train_features = preprocessing_features(file_train_images, all_features, all_descriptions)

    # TOKENIZER
    # give each word an index, and store that into tokenizer.p pickle file
    tokenizer = create_tokenizer(train_descriptions)
    vocab_size = len(tokenizer.word_index) + 1
    max_length = 32

    return all_descriptions, train_descriptions, train_features, tokenizer, vocab_size, max_length


