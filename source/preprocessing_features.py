from features import load_photos, disjoin_descriptions, select_features

#-----------------

def preprocessing_features(file_train_images, all_features, all_descriptions):
    train_imgs = load_photos(file_train_images)   
    train_descriptions = disjoin_descriptions(all_descriptions, train_imgs)
    train_features = select_features(all_features, train_imgs)
    return train_imgs, train_descriptions, train_features


#-----------------