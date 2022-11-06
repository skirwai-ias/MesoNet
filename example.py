# import numpy as np
from classifiers import *
# from pipeline import *
import streamlit as st
from PIL import Image
import os
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1 - Load the model and its pretrained weights
classifier = Meso4()
classifier.load('weights/Meso4_DF.h5')

# 2 - Minimial image generator
# We did use it to read and compute the prediction by batchs on test videos
# but do as you please, the models were trained on 256x256 images in [0,1]^(n*n)


st.title('Deepfake Detector')

# images = glob.glob("")

dataGenerator = ImageDataGenerator(rescale=1. / 255)

from PIL import Image
import glob


#
# btnResult = st.form_submit_button('Donald')
# if btnResult:
#
#     path = './test_images/donald_r/*.*'


def predict(res1):
    generator = dataGenerator.flow_from_directory(
        'test_images',
        target_size=(256, 256),
        batch_size=30,
        class_mode='binary',
        subset='training')

    st.image(res1, width=100)
    st.text('Ready for deeepfake prediction.......')
    num_to_label = {1: "real", 0: "fake"}

    # 3 - Predict
    X, y = generator.next()

    probabilistic_predictions = classifier.predict(X)

    print('Predicted :', probabilistic_predictions)

    predictions = [num_to_label[round(x[0])] for x in probabilistic_predictions]
    print(predictions)

    for ind, loc in enumerate(res1):
        st.image(loc, width=100)
        text = predictions[ind] + ' Probability is ' + str(probabilistic_predictions[ind][0])
        st.write(text)




option = st.selectbox(
    "How would you like to be contacted?",
    ("Choose dataset", "donald_r", "Donald", "girl_1","putin", "superman", "bruno"))
source_dir = ''
path = ''

print(option)

if option == 'donald_r':
    source_dir = f'datset/{option}'
    path = f'test_images/{option}'
    destination = f'test_images/{option}'
elif option == 'bruno':
    source_dir = f'datset/{option}'
    path = f'test_images/{option}'
    destination = f'test_images/{option}'
elif option == 'girl_1':
    source_dir = f'datset/{option}'
    path = f'test_images/{option}'
    destination = f'test_images/{option}'
elif option == 'Donald':
    source_dir = f'datset/{option}'
    path = f'test_images/{option}'
    destination = f'test_images/{option}'
elif option == 'putin':
    source_dir = f'datset/{option}'
    path = f'test_images/{option}'
    destination = f'test_images/{option}'
elif option == 'superman':
    source_dir = f'datset/{option}'
    path = f'test_images/{option}'
    destination = f'test_images/{option}'

if path != '' and source_dir != '':
    shutil.copytree(source_dir, destination)
    res = glob.glob(path + '/*.*')
    # dir_list = os.listdir(path)
    print(res)
    # image1 = Image.open(path + image)

    predict(res)
    shutil.rmtree(f'test_images/{option}')
    source_dir = ''
    path = ''
    option = ''

# 4 - Prediction for a video dataset

# classifier.load('weights/Meso4_F2F.h5')

# predictions = compute_accuracy(classifier, 'test_videos')
# for video_name in predictions:
#     print('`{}` video class prediction :'.format(video_name), predictions[video_name][0])
