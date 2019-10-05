# This script will train the FER model
from DexFERNet import *
from GenerateData import *
import argparse

"""parsing and configuration"""
def parse_args():
    desc = "...... Training FER ......"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--imagesize', type=int, default=48, help='The image resolution to be used')
    parser.add_argument('--opt_method', type=str, default='Adam', help='The optimizer to be used (options: Adam, Adamax, Nadam, SGD, RMSprop)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='The learning rate')
    parser.add_argument('--num_epoch', type=int, default=100, help='The epochs to train the model')
    parser.add_argument('--batchsize', type=int, default=256, help='The batch size')
    parser.add_argument('--model_path', type=str, default='model/FERModel.h5', help='The directory to save trained model')
    parser.add_argument('--dataset_type', type=str, default='csv', help='The dataset type to train the model (options: csv, image)')
    parser.add_argument('--root_path', type=str, default='dataset', help='The root path of dataset')
    parser.add_argument('--originaldatapath', type=str, default='dataset/original/fer2013.csv', help='Only set this when you use csv as dataset_type')
    return parser.parse_args()

def train_model(imagesize,
                opt_method,
                learning_rate,
                num_epoch,
                batchsize,
                model_path,
                dataset_type,
                root_path,
                originaldatapath):
    model = construct_network(imagesize)
    selected_optm = optimizer(learing_rate= learning_rate, method= opt_method)
    callbacks_list = [keras.callbacks.ModelCheckpoint(filepath=model_path,
                                                      monitor='val_acc',
                                                      save_best_only=True,
                                                      mode='max',
                                                      period=5), ]

    # LOAD DATASET: TRAINING, VALIDATION, AND TEST
    train_generator, val_generator, eval_generator, train_num, val_num, test_num = generate_data(dataset_type=dataset_type,
                                                                   root_path=root_path,
                                                                   img_size=imagesize,
                                                                   batch_size=batchsize,
                                                                   datapath=originaldatapath)
    # START TO TRAIN OUR MODEL
    compile_model(model,
                  selected_optm,
                  train_generator,
                  val_generator,
                  eval_generator,
                  num_epoch,
                  batchsize,
                  callbacks_list,
                  train_num,
                  val_num,
                  test_num,
                  val_tag=True)


if __name__ == '__main__':
    args = parse_args()
    if args is None:
        exit()
    imagesize = args.imagesize
    opt_method = args.opt_method
    learning_rate = args.learning_rate
    num_epoch = args.num_epoch
    batchsize = args.batchsize
    model_path = args.model_path
    dataset_type = args.dataset_type
    root_path = args.root_path
    originaldatapath= args.originaldatapath

    train_model(imagesize,
                opt_method,
                learning_rate,
                num_epoch,
                batchsize,
                model_path,
                dataset_type,
                root_path,
                originaldatapath)