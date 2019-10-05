from matplotlib import pyplot
from numpy import load
from numpy.random import randint
# plot a list of loaded faces
def plot_image(images, n, save_name):
	for i in range(n * n):
		# define subplot
		pyplot.subplot(n, n, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(images[i].astype('uint8'))
	pyplot.savefig(save_name)

def generate_samples(dataset_name, save_name, sample_number):
    data = load(dataset_name)
    cities = data['arr_0']
    print('Loaded: ', cities.shape)
    ix = randint(0, cities.shape[0], sample_number * sample_number)
    # select images
    target = cities[ix]
    plot_image(target, sample_number, save_name)

if __name__ == '__main__':
    dataset_name = 'dataset/preprocessing_result/city_64.npz'
    save_name = 'use_results/original_samples.png'
    sample_number_in_each_row = 5 # attention here the
    generate_samples(dataset_name, save_name, sample_number_in_each_row)
    print('Finish')