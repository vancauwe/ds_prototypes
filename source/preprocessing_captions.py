from captions import all_img_captions, cleaning_text, join_descriptions, text_vocabulary

# --------------------------------------------------

def preprocessing_captions(text_file):
    descriptions = all_img_captions(text_file)
    #print("Length of descriptions =" ,len(descriptions))
    #cleaning the descriptions
    clean_descriptions = cleaning_text(descriptions)
    #building vocabulary 
    vocabulary = text_vocabulary(clean_descriptions)
    #print("Length of vocabulary = ", len(vocabulary))
    all_descriptions = join_descriptions(clean_descriptions)
    return descriptions, clean_descriptions, vocabulary, all_descriptions

# --------------------------------------------------

