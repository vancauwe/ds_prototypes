def load_photos(file):
    photos = file.split("\n")[:-1]
    return photos

def select_features(all_features, photos):
    train_features = {k:all_features[k] for k in photos}
    return train_features


def disjoin_descriptions(all_descriptions, photos): 
    #loading clean_descriptions
    descriptions = {}
    for line in all_descriptions.split("\n"):
        words = line.split()
        if len(words)<1 :
            continue
        image, image_caption = words[0], words[1:]
        if image in photos:
            if image not in descriptions:
                descriptions[image] = []
            desc = '<start> ' + " ".join(image_caption) + ' <end>'
            descriptions[image].append(desc)
    return descriptions