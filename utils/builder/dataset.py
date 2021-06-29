import os
from utils.mfcc.feature import SpeechFeature


def dataSetBuilder(dir):
    fileList = []
    dataset = {}
    #Filter the .wav files in the folder
    for f in os.listdir(dir):
        if os.path.splitext(f)[1] == ".wav":
            fileList.append(f)

    # Extract only the labels
    for file in fileList:
        tmp = file.split('.')[0]
        str = ''.join(tmp)
        _str = str.rstrip('0123456789')
        label = _str
        feature = SpeechFeature(dir+file)
        
        # Check if labeled word exists in array
        if label not in dataset.keys():
            dataset[label] = []
            # append feature to label
            dataset[label].append(feature)
        else:
            exist_feature = dataset[label]
            exist_feature.append(feature)
            dataset[label] = exist_feature
        
    return dataset