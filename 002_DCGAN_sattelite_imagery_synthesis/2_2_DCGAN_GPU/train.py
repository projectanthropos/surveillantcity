from DCGAN import *
from util import *
from imagePreProcessing import *
import time
import argparse

# THIS FUNCTION DEFINES THE ARGUMENTS WILL BE USED IN THE API
# IF NOT GIVEN A PARTICULAR VALUE, IT WILL USE THE DEFAULT VALUE
"""parsing and configuration"""
def parse_args():
    desc = "...... Training DCGAN ......"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--image_dir', type=str, default='dataset/images/', help='The dataset location')
    parser.add_argument('--inputfile_dir', type=str, default='dataset/preprocessing_result/city_64.npz', help='The preprocessing dataset location')
    parser.add_argument('--IMAGE_RESO', type=int, default= 64, help='The image resolution')
    parser.add_argument('--latent_dim', type=int, default=100, help='The latent sapce vector size')
    parser.add_argument('--n_epochs', type=int, default=10000, help='The training epoch')
    parser.add_argument('--n_batch', type=int, default=32, help='The batch size')
    return parser.parse_args()

if __name__ == '__main__':
    start = time.time()
    # parse arguments
    args = parse_args()
    if args is None:
        exit()
    filename = args.inputfile_dir
    image_dir = args.image_dir
    IMAGE_RESO = args.IMAGE_RESO
    if IMAGE_RESO%4 !=0:
        print('IMAGE_RESO has to be a multiple of 4!')
        print('Please change the input and try again!')
        exit()
    main(image_dir, filename, IMAGE_RESO)
    dataset = load_real_samples(filename)
    latent_dim = args.latent_dim
    n_epochs = args.n_epochs
    n_batch = args.n_batch
    GENERATE_RES = int(int(IMAGE_RESO/4) ** 0.5)
    #GENERATE_RES = 2 # produce image resolution: (4 * GENERATE_RES^2) * (4 * GENERATE_RES^2)
    g_model = make_generator_model(latent_dim, GENERATE_RES)
    d_model = make_discriminator_model(image_shape =[64,64,3], GENERATE_RES = GENERATE_RES)
    gan_model = make_dcgan(g_model, d_model)
    train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs, n_batch)
    elapsed = (time.time() - start)
    print("Time used:", elapsed)
    pass
