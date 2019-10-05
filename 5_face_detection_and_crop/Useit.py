# THIS SCRIPT IS THE USAGE API FOR FACE DETECTOR & CROP

# IMPORT PACKAGES
import argparse
import os
import glob
import face_detection_crop as fdc

# THIS FUNCTION DEFINES THE ARGUMENTS WILL BE USED IN THE API
# IF NOT GIVEN A PARTICULAR VALUE, IT WILL USE THE DEFAULT VALUE
"""parsing and configuration"""
def parse_args():
    desc = "...... The API of Face Detector and Crop ......"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--scale_factor', type=float, default=1.6, help='How much the image size is reduced at each image scale ')
    parser.add_argument('--min_neighbors', type=int, default=6, help='How many neighbors each candidate rectangle should have to retain it')
    parser.add_argument('--min_size_w', type=int, default=1, help='Minimum possible object size (width).')
    parser.add_argument('--min_size_h', type=int, default=1, help='Minimum possible object size (height).')
    parser.add_argument('--save_path_d', type=str, default='img_video_detector_output/', help='The path to save detector results')
    parser.add_argument('--save_path_c', type=str, default='img_video_crop_output/', help='The path to save crop results')
    parser.add_argument('--showflag', type=bool, default=True, help='True or False')
    parser.add_argument('--videocropflag', type=bool, default=True, help='True or False')
    return check_args(parser.parse_args())


# CHECK ARGUMENTS TO MEET PARTICULAR REQUIREMENTS
"""checking arguments"""
def check_args(args):
    check_folder(args.save_path_d)
    check_folder(args.save_path_c)
    return args

# CHECK WHETHER THIS DIRECTORY IS EXIST OR NOT;
# IF NOT EXIST, THE CORRESPONDING DIRECTORY WILL BE CREATED.
def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    facedc = fdc.FaceDetectCrop(args.scale_factor,
                                args.min_neighbors,
                                args.min_size_w,
                                args.min_size_h,
                                args.save_path_d,
                                args.save_path_c,
                                args.showflag,
                                args.videocropflag)
    # test image path
    testfile_images = 'images_input/*'
    images =  glob.glob(testfile_images)
    for image_name in images:
        facedc.face_detector_image(image_name)

    # test video path

    testfile_videos = 'videos_input/*'
    videos = glob.glob(testfile_videos)
    for video in videos:
        facedc.face_detector_video(video)

    print('All results have been saved!')



if __name__ == '__main__':
    main()
    pass
