**Implementation/Documentation/review by Taihui Li, research work under the supervision of Vahan M. Misakyan**

&nbsp;
&nbsp;


# Satellite Imagery

## Table of Contents

1. [Environment Setting Up](#1-environment-setting-up)<br>
  1.1 [Required Dependencies](#11-required-dependencies)<br>
  1.2 [Installation Guide](#12-installation-guide)<br>
2. [Scripts/Directories Introduction](#2-scriptsdirectories-introduction)
3. [Usage](#3-usage)
4. [Satellite Imagery Dataset](#4-satellite-imagery-dataset)


## 1 Environment Setting Up
In order to run this program, you need to set up your exectution environment approparirately. Furthermore, you might not willing to re-install your computer system when somethings go wrong. It is therefore strongly recommended to set up a [Virtual Environment](https://www.geeksforgeeks.org/python-virtual-environment/) for each program. Fortunately, the [Anaconda](https://www.anaconda.com/distribution/) makes life much easier.

### 1.1 Required Dependencies
* [Python3.6](https://www.python.org/download/releases/3.0/).
* [Mapbox](https://www.mapbox.com/).

### 1.2 Installation Guide
1. Download Anaconda [here](https://www.anaconda.com/distribution/) and follow the [official guidance](https://docs.anaconda.com/anaconda/install/) to install Anaconda.

2. Create an Virtual Environment for this program. For example, I named my program as ```SatelliteImagery```, I will create it by using the following command: (more details can found [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands))
```
conda create -n SatelliteImagery python=3.6
```

3. List the Virtual Environment in your laptop:
```
conda info --envs
```

4. Activate the desired Virtual Environment (here is ```SatelliteImagery```):
```
source activate SatelliteImagery
```
Or you can use:
```
conda activate SatelliteImagery
```

5. Install ```mapbox```:
```
conda install --channel "mapbox" package
```

6. Visti ```mapbox``` website to get the [Access tokens
](https://docs.mapbox.com/help/how-mapbox-works/access-tokens/).

7. Deactivate your virtual environment:
```
source deactivate SatelliteImagery
```
Or you can use:
```
conda deactivate SatelliteImagery
```


## 2 Scripts/Directories Introduction
This section introduces the scripts and directories in this implement code.

* **parseGeoJSON.py**: It's the core of getting satellite images. It will parse the GeoJSON docuemnts and convert the records to images.

## 3 Usage
1. Activate your Virtal Environment:
```
source activate SatelliteImagery
```
Or you can use:
```
conda activate SatelliteImagery
```

2. Swith to the directory where the exectution program is located:
```
cd ~/home/PycharmPrograms/SatelliteImagery/
```

3. Run the program (note you should replace ```GeoJSON_DIR``` with the exact location where you store your GeoJSON files):
```
export MAPBOX_ACCESS_TOKEN="Your_ACCESS_TOKEN_HERE" && python parseGeoJSON.py GeoJSON_DIR
```
Note, you must replace ```Your_ACCESS_TOKEN_HERE``` with the correct one you got from Mapbox.


## 4 Satellite Imagery Dataset
There is a GitHub website where you can find a lot of Satellite Imagery information/dateset. More details should visit [this website](https://github.com/chrieke/awesome-satellite-imagery-datasets).








