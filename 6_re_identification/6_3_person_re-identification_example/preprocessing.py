# This script will preprocessing dataset

# import packages
import os
from shutil import copyfile
import argparse

# define a class to process raw dataset
class PreProcess:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir

    # generate the related directory
    def makedir(self):
        save_path = os.path.join(self.dataset_dir, 'pytorch')
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        # query save path
        query_save_path = os.path.join(save_path, 'query')
        if not os.path.isdir(query_save_path):
            os.mkdir(query_save_path)

        # multi-query path
        multi_query_save_path = os.path.join(save_path, 'multi-query')
        if not os.path.isdir(multi_query_save_path):
            os.mkdir(multi_query_save_path)

        # gallery path
        gallery_save_path = os.path.join(save_path, 'gallery')
        if not os.path.isdir(gallery_save_path):
            os.mkdir(gallery_save_path)

        # train all path
        train_all_save_path = os.path.join(save_path, 'train_all')
        if not os.path.isdir(train_all_save_path):
            os.mkdir(train_all_save_path)

        # train save path
        train_save_path = os.path.join(save_path, 'train')
        if not os.path.isdir(train_save_path):
            os.mkdir(train_save_path)

        # eval save path
        val_save_path = os.path.join(save_path,'val')
        if not os.path.isdir(val_save_path):
            os.mkdir(val_save_path)

        dir_list = []
        dir_list.append(query_save_path)
        dir_list.append(multi_query_save_path)
        dir_list.append(gallery_save_path)
        dir_list.append(train_all_save_path)
        dir_list.append(train_save_path)
        dir_list.append(val_save_path)
        return dir_list

    # the function to process data
    def process(self):
        dir_list = self.makedir()
        query_save_path = dir_list[0]
        multi_query_save_path = dir_list[1]
        gallery_save_path = dir_list[2]
        train_all_save_path = dir_list[3]
        train_save_path = dir_list[4]
        val_save_path = dir_list[5]

        # -----------------------------------------
        # query
        query_path = os.path.join(self.dataset_dir, 'query')
        for root, dirs, files in os.walk(query_path, topdown=True):
            for name in files:
                if not name[-3:] == 'jpg':
                    continue
                ID = name.split('_')
                src_path = query_path + '/' + name
                dst_path = query_save_path + '/' + ID[0]
                if not os.path.isdir(dst_path):
                    os.mkdir(dst_path)
                copyfile(src_path, dst_path + '/' + name)

        # -----------------------------------------
        # multi-query
        query_path = self.dataset_dir + '/gt_bbox'
        # for dukemtmc-reid, we do not need multi-query
        if os.path.isdir(query_path):
            for root, dirs, files in os.walk(query_path, topdown=True):
                for name in files:
                    if not name[-3:] == 'jpg':
                        continue
                    ID = name.split('_')
                    src_path = query_path + '/' + name
                    dst_path = multi_query_save_path + '/' + ID[0]
                    if not os.path.isdir(dst_path):
                        os.mkdir(dst_path)
                    copyfile(src_path, dst_path + '/' + name)

        # -----------------------------------------
        # gallery
        gallery_path = self.dataset_dir + '/bounding_box_test'
        for root, dirs, files in os.walk(gallery_path, topdown=True):
            for name in files:
                if not name[-3:] == 'jpg':
                    continue
                ID = name.split('_')
                src_path = gallery_path + '/' + name
                dst_path = gallery_save_path + '/' + ID[0]
                if not os.path.isdir(dst_path):
                    os.mkdir(dst_path)
                copyfile(src_path, dst_path + '/' + name)

        # ---------------------------------------
        # train_all
        train_path = self.dataset_dir + '/bounding_box_train'
        for root, dirs, files in os.walk(train_path, topdown=True):
            for name in files:
                if not name[-3:] == 'jpg':
                    continue
                ID = name.split('_')
                src_path = train_path + '/' + name
                dst_path = train_all_save_path + '/' + ID[0]
                if not os.path.isdir(dst_path):
                    os.mkdir(dst_path)
                copyfile(src_path, dst_path + '/' + name)


        # ---------------------------------------
        # train_val
        train_path = self.dataset_dir + '/bounding_box_train'
        for root, dirs, files in os.walk(train_path, topdown=True):
            for name in files:
                if not name[-3:] == 'jpg':
                    continue
                ID = name.split('_')
                src_path = train_path + '/' + name
                dst_path = train_save_path + '/' + ID[0]
                if not os.path.isdir(dst_path):
                    os.mkdir(dst_path)
                    dst_path = val_save_path + '/' + ID[0]  # first image is used as val image
                    os.mkdir(dst_path)
                copyfile(src_path, dst_path + '/' + name)



def parse_args():
    desc = "...... PreProcessing Images ......"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--dataset_dir', type=str, default='dataset/Market/', help='The directory of dataset')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args is None:
        exit()
    dataset_dir = args.dataset_dir
    if not os.path.isdir(dataset_dir):
        print("Please change the directory path to your dataset directory!")
        exit()
    preprocess_obj = PreProcess(dataset_dir)
    preprocess_obj.process()
    print("Preprocessing has been done!")
    pass
