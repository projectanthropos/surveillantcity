# example of extracting and resizing faces into a new dataset
from numpy import asarray
from numpy import savez_compressed
from PIL import Image
from matplotlib import pyplot
from numpy import load
import glob
import os



# load an image as an rgb numpy array
def load_image(image_name,IMAGE_RESO):
    try:
        image = Image.open(image_name)
    except:
        print('This image cannot be open！')
        return
    if image.mode != 'RGB':
        image = image.convert("RGB")
    # convert from integers to floats
    image = image.resize((IMAGE_RESO, IMAGE_RESO), Image.ANTIALIAS)
    pixels = asarray(image)
    pixels = pixels.astype('float32')
    return pixels

def save_dataset(image_dir,save_name, IMAGE_RESO):
    if os.path.isfile(save_name):
        print("Loading previous training pickle...")
        return
    image_dir = os.path.join(image_dir,'*')
    images = glob.glob(image_dir)
    save_result = list()
    for image_name in images:
        save_result.append(load_image(image_name,IMAGE_RESO))
        print(len(save_result), load_image(image_name,IMAGE_RESO).shape)
    save_result = asarray(save_result)
    print('Loaded: ', save_result.shape)
    savez_compressed(save_name, save_result)
    print('All data has been saved successfully!')

# plot a list of loaded faces
def plot_image(images, n):
	for i in range(n * n):
		# define subplot
		pyplot.subplot(n, n, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(images[i].astype('uint8'))
	pyplot.show()

def main(image_dir, save_name, IMAGE_RESO = 64):
    save_dataset(image_dir,save_name, IMAGE_RESO)
    data = load(save_name)
    cities = data['arr_0']
    print('Loaded: ', cities.shape)
    plot_image(cities, 2)

if __name__ == '__main__':
    # directory that contains all images
    directory = 'dataset/images/*'
    save_name = 'dataset/preprocessing_result/city_64.npz'
    IMAGE_RESO = 64
    save_dataset(directory)
