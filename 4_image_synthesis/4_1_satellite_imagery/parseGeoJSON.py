# THIS SCRIPT WILL PARSE THE GEOJSON FILE AND RETURN THE RECORDS
# IMPORT PACKAGES
import glob
from mapbox import Static
import os
import json
import argparse


class ParseGeojson:
    def __init__(self, filedir, count):
        self.filedir = filedir
        self.count = count

    # define a function to get the images from GeoJSON
    def get_images(self):
        files = glob.glob(self.filedir)
        for file in files:
            with open(file,'r') as cur_file:
                alldata = json.load(cur_file)
            for feature in alldata['features']:
                try:
                    geojson_str = feature
                    image_name = 'images/' + str(self.count) + '.png'
                    self.transfer2image(geojson_str, image_name)
                    self.count += 1
                    print('@@@:' + str(self.count) + 'images have been converted successfuly!\n')
                except:
                    print('This image cannot be converted!\n')

    # define a function to convert GeoJSON to images
    def transfer2image(self, geojson_str, image_name):
        service = Static()
        service.session.params['access_token'] == os.environ['MAPBOX_ACCESS_TOKEN']
        response = service.image('mapbox.satellite', features=[geojson_str])
        print(response.status_code)
        # add to a file
        with open(image_name, 'wb') as output:
            _ = output.write(response.content)
        print('Current image has been saved!')


"""parsing and configuration"""
def parse_args():
    desc = "...... The API of Statellite Imagery ......"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('geojsonDir', type=str, help='The input GeoJSON directory')
    return parser.parse_args()

if __name__ == '__main__':
    # check the "done" file
    input_args = parse_args()
    if os.path.exists("done.txt"):
        pass
    filedir = os.path.join(input_args.geojsonDir, '*')
    print(filedir)
    pgjson = ParseGeojson(filedir, 0)
    pgjson.get_images()
    pass



