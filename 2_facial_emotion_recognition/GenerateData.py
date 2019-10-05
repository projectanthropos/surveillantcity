# THIS SCRIPT WILL GENERATE THE DATE WE NEED TO TRAIN OUR NEURAL NETWORK
# THE DATASET CAN BE DOWNLOADED VIA THIS LINK:
# https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

# IMPORT PACKAGES
import pandas as pd
import os
import numpy as np
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator


#==========================================
# Name: generate_image
# Purpose: TRANSFORM IMAGES DATA FROM PANDAS/DATAFRAME INTO IMAGES
# Input Parameter(s): df --- THE DATAFRAME CONTAINS IMAGE DATA
#                     savepath --- THE PATH WE WANT TO SAVE OUR IMAGES
# Return Value(s): NONE
#============================================
def generate_image(df, savepath, imagesize):
    # USE FOR LOOP TO SCAN EACH RECORD IN DATAFRAME
    for trn_index in range(0, len(df)):
        record = df.iloc[trn_index]
        # 'emotion' IS THE LABEL FOR OUR EMOTION, IT RANGES FROM 0 TO 6
        # 'pixels' IS THE DATA INFORMATION FOR OUR IMAGE
        emotion = record['emotion']
        pixels = record['pixels']
        pixels_list = []
        for p in pixels.split():
            pixels_list.append(float(p))
        pixels = np.asarray(pixels_list).reshape(imagesize, imagesize)
        subfolder = os.path.join(savepath, str(emotion))
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
        im = Image.fromarray(pixels).convert('L')
        image_name = os.path.join(subfolder, '{:05d}.jpg'.format(trn_index))
        im.save(image_name)
        print('Image:' + image_name + ' has been saved!\n')

#==========================================
# Name: separate_data
# Purpose: SEPARATE THE ORIGINAL CSV DATA INTO TRAINING, VALIDATION, AND TEST IMAGES DATASETS
# Input Parameter(s): datapath --- THE PATH OF THE ORIGINAL CSV
# Return Value(s): NONE
#============================================
def separate_data(imagesize, datapath = 'dataset/original/fer2013.csv'):
    # SEPARATE THE DATESET INTO TRAINING, VALIDATION, AND TEST
    dataset = pd.read_csv(datapath)
    trn_df = dataset[dataset['Usage'] == 'Training']
    val_df = dataset[dataset['Usage'] == 'PublicTest']
    tst_df = dataset[dataset['Usage'] == 'PrivateTest']

    # CREATE SAVING PATH
    trn_path = os.path.join('dataset/', 'train')
    val_path = os.path.join('dataset/', 'val')
    tst_path = os.path.join('dataset/', 'test')

    for savepath in [trn_path, val_path, tst_path]:
        if not os.path.exists(savepath):
            os.makedirs(savepath)

    # CALL FUNCTION TO SAVE IMAGES
    generate_image(trn_df, trn_path,imagesize)
    generate_image(val_df, val_path,imagesize)
    generate_image(tst_df, tst_path,imagesize)

    print('Congratulations! All images have been saved!\n')



#==========================================
# Name: generate_data
# Purpose: GENERATE DATA THAT IS SUITABLE FOR OUR NEURAL NETWORK
# Input Parameter(s): root_path --- THE DIRECTORY WHERE THE TRAINING, VALIDATION, AND TEST DATA RESIDED
#                     img_size --- THE SIZE THE IMAGE SHOULD BE (DEFAULT 48)
#                     batch_size --- THE BATCH SIZE (DEFAULT 256)
# Return Value(s): train_generator --- TRAINING DATA SET
#                  val_generator --- VALIDATION DATA SET
#                  eval_generator --- TEST DATA SET
#============================================
def generate_data(dataset_type='csv', root_path='dataset', img_size=48, batch_size=256, datapath = 'dataset/original/fer2013.csv'):
    if dataset_type == 'csv':
        separate_data(img_size, datapath)
    # TRAINING DATA
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    train_generator = train_datagen.flow_from_directory(
        root_path + '/train',
        target_size=(img_size, img_size),
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical')

    # VALIDATION DATA
    val_datagen = ImageDataGenerator(
        rescale=1. / 255)
    val_generator = val_datagen.flow_from_directory(
        root_path + '/val',
        target_size=(img_size, img_size),
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical')

    # TEST DATA
    eval_datagen = ImageDataGenerator(
        rescale=1. / 255)
    eval_generator = eval_datagen.flow_from_directory(
        root_path + '/test',
        target_size=(img_size, img_size),
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical')

    train_num = count_images(root_path + '/train/')
    val_num = count_images(root_path + '/val/')
    test_num = count_images(root_path + '/test/')
    return train_generator, val_generator, eval_generator, train_num, val_num, test_num


def count_images(target_path):
    file_count = 0
    for (root, dirs, files) in os.walk(target_path, topdown=True):
        file_count = file_count + len(files)
    return file_count


if __name__ == '__main__':
    generate_data()