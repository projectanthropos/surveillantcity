# THIS SCRIPT WILL USE THE MODEL WE TRAINED TO RECOGNITION EMOTION FROM PICTURES
from keras.models import load_model
import numpy as np
import cv2
from keras.preprocessing import image
import os
import glob
import argparse



# Import the xml files of frontal face and eye
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

objects = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']




#==========================================
# Name: get_emotion
# Purpose: CALL MODEL TO MAKE PREDICTION
#          RETURN THE PROBABILITY FOR EACH EMOTION
# Input Parameter(s): modelpath --- THE PATH OF TRAINED MODEL
#                     tstimage --- THE IMAGE WE WANT TO TEST
# Return Value(s): result --- A PROBABILITY LIST FOR EACH EMOTION
#============================================
def get_emotion(modelpath, tstimage, imagesize):
    # RESIZE IMAGE TO 48*48 AS WE USED THIS SIZE TO TRAIN OUR MODEL
    img = cv2.resize(tstimage, (imagesize, imagesize), interpolation=cv2.INTER_CUBIC)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255
    # LOAD OUR MODEL
    model = load_model(modelpath)
    # MODEL PREDICTION
    result = model.predict(x)
    return result


#==========================================
# Name: face_detect
# Purpose: DETECT FACES IN IMAGES (USING Viola Jones)
# Input Parameter(s): image_path --- THE PATH OF IMAGES
# Return Value(s): faces --- FACES DETECTED BY OUR PROGRAM
#                  img_gray --- THE GRAY IMAGE
#                  img --- THE ORIGINAL IMAGE
#============================================
def face_detect(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # LOAD THE IMAGE AND CONVERT IT TO BGRGRAY
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # DETECT FACES
    faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)
    return faces, img_gray, img


#==========================================
# Name: display_image
# Purpose: CALL FUNCTIONS DEFINED IN THIS SCRIPT,
#          SHOW THE RECOGNITION RESULTS AND SAVE RESULTS
# Input Parameter(s): modelpath --- THE PATH OF OUR TRAINED MODEL
#                     imagepath --- THE PATH OF IMAGES
# Return Value(s): NONE
#============================================
def display_image(modelpath, imagepath, imagesize=48, video_flag = False):
    result_dir = 'test_and_result/result_images/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    current_image_name = imagepath.split('/')[-1]
    result_image_name = 'result_' + current_image_name
    result_image_name = os.path.join(result_dir, result_image_name)

    faces, img_gray, img = face_detect(imagepath)
    spb = img.shape
    sp = img_gray.shape
    height = sp[0]
    width = sp[1]
    size = 600
    emo = ""
    face_exists = 0

    for (x, y, w, h) in faces:
        face_exists = 1
        face_img_gray = img_gray[y:y + h, x:x + w]
        result = get_emotion(modelpath, face_img_gray, imagesize)
        result = list(result[0])
        print('Angry:', result[0], 'Disgust:', result[1], ' Fear:', result[2], ' Happy:', result[3], ' Sad:', result[4],
              ' Surprise:', result[5], ' Neutral:', result[6])
        label = result.index(max(result))
        emo = objects[label]
        t_size = 2
        ww = int(spb[0] * t_size / 300)
        www = int((w + 10) * t_size / 100)
        www_s = int((w + 20) * t_size / 100) * 2 / 5
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), ww)
        cv2.putText(img, emo, (x + 2, y + h - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        www_s, (255, 0, 255), thickness=www, lineType=1)
    if face_exists:
        if video_flag == False:
            cv2.HoughLinesP
            cv2.namedWindow(emo, 0)
            cent = int((height * 1.0 / width) * size)
            cv2.resizeWindow(emo, (size, cent))

            cv2.imshow(emo, img)
            cv2.imwrite(result_image_name, img)
            print('Press any key to close current window!')
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return img



# video process
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html#saving-a-video
# THIS FUNCTION CONDUCTS FACE DETECTION;
# IT WILL DRAW A RECTANGLE AROUND FACES; ALSO, CALL FACE CROP FUNCTION
# video_name --- IS THE PROCESSING VIDEO
def video_process(modelpath, video_name, imagesize):
    result_dir = 'test_and_result/result_videos/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    save_name = video_name.split('/')[-1]
    save_name = save_name.split('.')[0]
    save_name = 'result_' + save_name + '.avi'
    final_path = os.path.join(result_dir, save_name)
    input_movie = cv2.VideoCapture(video_name)
    #input_movie.open(video_name)
    fps = input_movie.get(cv2.CAP_PROP_FPS)
    frames = input_movie.get(cv2.CAP_PROP_FRAME_COUNT)
    infourcc = input_movie.get(cv2.CAP_PROP_FPS)
    framesize_w = int(input_movie.get(cv2.CAP_PROP_FRAME_WIDTH))
    framesize_h = int(input_movie.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
    output_movie = cv2.VideoWriter(final_path, fourcc, fps, (framesize_w, framesize_h), isColor=1)
    frameid = 0
    while True:
        # Grab a single frame of video
        success, frame = input_movie.read()
        #### write it into disk and then return its path
        cv2.imwrite('test_and_result/test_videos/temp.png', frame)
        frame_path = 'test_and_result/test_videos/temp.png'
        frameid += 1
        # Quit when the input video file ends
        if not success:
            break
        ##########################
        ##### emotion detection for a frame of video
        result_frame = display_image(modelpath, frame_path, imagesize, True)

        #### delete the frame on disk
        if os.path.exists(frame_path):
            os.remove(frame_path)
        print("Writing frame {} / {}".format(frameid, frames))
        output_movie.write(result_frame)
    # FINISH
    input_movie.release()




"""parsing and configuration"""
def parse_args():
    desc = "...... The Usage API of FER ......"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--modelpath', type=str, default='model/FERModel.h5', help='The FER model to be used')
    parser.add_argument('--imagepath', type=str, default='test_and_result/test_images/', help='The images directory')
    parser.add_argument('--videopath', type=str, default='test_and_result/test_videos/', help='The videos directory')
    parser.add_argument('--imagesize', type=int, default=48, help='The size of image')
    parser.add_argument('--type_tag', type=int, default=0, help='0:images; 1: videos')
    return parser.parse_args()





if __name__ == '__main__':
    args = parse_args()
    if args is None:
        exit()
    modelpath = args.modelpath
    imagepath = args.imagepath
    videopath = args.videopath
    imagesize = args.imagesize
    type_tag = args.type_tag

    if type_tag == 0:
        imagepath = os.path.join(imagepath, '*')
        images = glob.glob(imagepath)
        for image_name in images:
            display_image(modelpath, image_name, imagesize)

    if type_tag == 1:
        videopath = os.path.join(videopath, '*')
        videos = glob.glob(videopath)
        for video in videos:
            video_process(modelpath, video, imagesize)

    """
    if len(sys.argv) != 2:
        print('The numbers of arguments are not correct!\n')
        print('Following this instruction:\n')
        print('python EmotionRec.py Your_Images_Dir\n')
    else:
        modelpath = 'model/FERModel.h5'
        imagepath = sys.argv[1]
        for image_name in os.listdir(imagepath):
            display_image(modelpath, imagepath)
    """



