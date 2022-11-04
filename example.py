# import numpy as np
from classifiers import *
# from pipeline import *

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1 - Load the model and its pretrained weights
classifier = Meso4()
classifier.load('weights/Meso4_DF.h5')

# 2 - Minimial image generator
# We did use it to read and compute the prediction by batchs on test videos
# but do as you please, the models were trained on 256x256 images in [0,1]^(n*n)

dataGenerator = ImageDataGenerator(rescale=1. / 255)

generator = dataGenerator.flow_from_directory(
    'test_images',
    target_size=(256, 256),
    batch_size=30,
    class_mode='binary',
    subset='training')

num_to_label = {1: "real", 0: "fake"}

# 3 - Predict
X, y = generator.next()


probabilistic_predictions = classifier.predict(X)

print('Predicted :', probabilistic_predictions)

predictions = [num_to_label[round(x[0])] for x in probabilistic_predictions]
print(predictions)

# 4 - Prediction for a video dataset

# classifier.load('weights/Meso4_F2F.h5')

# predictions = compute_accuracy(classifier, 'test_videos')
# for video_name in predictions:
#     print('`{}` video class prediction :'.format(video_name), predictions[video_name][0])
