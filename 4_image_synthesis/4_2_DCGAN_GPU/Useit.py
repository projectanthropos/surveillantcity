# USE THE GENERATOR MODEL AND GENERATE NEW IAMGES

# IMPORT PACKAGES
from keras.models import load_model
from matplotlib import pyplot
import util
import imageio
import argparse
import os


# SET UP VARIABLES
# THIS FUNCTION DEFINES THE ARGUMENTS WILL BE USED IN THE API
# IF NOT GIVEN A PARTICULAR VALUE, IT WILL USE THE DEFAULT VALUE
"""parsing and configuration"""
def parse_args():
    desc = "...... The API of DCGAN ......"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--model_name', type=str, default='training_results/models/generator_model_650.h5', help='The model to use')
    parser.add_argument('--video_name', type=str, default='use_results/gan.mp4', help='The result video name')
    parser.add_argument('--video_name_training', type=str, default='use_results/gan_training_process.mp4', help='The result video name')
    parser.add_argument('--plot_name', type=str, default='use_results/gan.png', help='The result plot name')
    parser.add_argument('--n_samples', type=int, default=25, help='The number of samples want to generate')
    parser.add_argument('--row_col', type=int, default=5, help='The plot rows and columns')
    parser.add_argument('--latent_dim', type=int, default=100, help='The latent sapce vector size')
    return parser.parse_args()

# plot the generated images
# examples --- THE GENERATED IMAGES
# n --- THE PLOT GRID ROW AND COLUMN
# plot_name --- THE NAME TO BE SAVED
def create_plot(examples, n, plot_name):
    # plot images
    for i in range(n * n):
        # define subplot
        pyplot.subplot(n, n, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        try:
            imageio.imwrite('use_results/result_'+str(i+1)+'.png',examples[i])
        except:
            print('This image cannot be saved!')
        pyplot.imshow(examples[i, :, :])
    pyplot.savefig(plot_name)
    pyplot.show()

# THIS FUNCTION WILL CREATE A VIDEO TO SHOW THE GENERATED RESULTS
# examples --- THE GENERATED IMAGES
# video_name --- THE NAME TO BE SAVED
def create_video(examples, video_name):
    imageio.mimsave(video_name, examples, fps=1000)

def create_video_training(video_name_training):
    examples = os.listdir('training_results/plots/')
    imageio.mimsave(video_name_training, examples, fps=1000)

# THIS FUNCTION CONDUCTS THE GENERATION
# model_name --- THE TRAINED MODEL TO BE USED
# latent_dim --- THE INPUT VECTOR SIZE
# n_samples --- THE NUMBER OF IMAGES WANT TO GENERATE
# row_col --- THE PLOT GRID ROW AND COLUMN
# plot_name --- THE PLOT NAME TO BE SAVED
# video_name --- THE VIDEO NAME TO BE SAVED
def usage(model_name, latent_dim, n_samples, row_col, plot_name, video_name, video_name_training):
    # load model
    model = load_model(model_name)
    # generate images
    latent_points = util.generate_latent_points(latent_dim, n_samples)
    X = model.predict(latent_points)
    # scale from [-1,1] to [0,1]
    X = (X + 1) / 2.0
    # save it as video
    create_video(X, video_name)
    # plot the result
    create_plot(X, row_col,plot_name)
    # save the training processing video
    create_video_training(video_name_training)

if __name__ == '__main__':
    # parse arguments
    args = parse_args()
    if args is None:
        exit()
    model_name = args.model_name
    n_samples = args.n_samples
    row_col = args.row_col
    plot_name = args.plot_name
    video_name = args.video_name
    video_name_training = args.video_name_training
    latent_dim = args.latent_dim
    usage(model_name, latent_dim, n_samples, row_col, plot_name, video_name, video_name_training)
    pass
