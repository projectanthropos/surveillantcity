**Implementation/Documentation/review by Taihui Li, research work under the supervision of Vahan M. Misakyan**

&nbsp;
&nbsp;


# Person Re-Identification Implementation

This repo works as a practice example/demo to Person Re-Identification (ReID). The code credits to this great [tutorial](https://github.com/layumi/Person_reID_baseline_pytorch) (some modifications have been made).



## Table of Contents

1. [Environment Setting Up](#1-environment-setting-up)<br>
    1.1 [Required Dependencies](#11-required-dependencies)<br>
    1.2 [Installation Guide](#12-installation-guide)<br>
    1.3 [Get Dataset](#13-get-dataset)<br>  
2. [Scripts/Directories Introduction](#2-scriptsdirectories-introduction)
3. [Usage](#3-usage)<br>
    3.1 [Preprocessing Data](#31-preprocessing-data)<br>
    3.2 [Construct Neural Network Model](#32-construct-neural-network-model)<br>
    3.3 [Training](#33-training)<br>
    3.4 [Test](#34-test)<br>
    3.5 [Evaluation](#35-evaluation)<br>
    3.6 [Visualization](#36-visualization)<br> 
4. [Further Reading](#4-further-reading)
5. [Reference](#reference)





## 1 Environment Setting Up

### 1.1 Required Dependencies

* Anaconda 3
* Python 3.6
* PyTorch 
* TorchVision
* apex
* scipy
* matplotlib
* yaml



### 1.2 Installation Guide

1. Create a virtual environment named ```ReID``` (the benefit of using virtual environment can be found [here](https://www.geeksforgeeks.org/python-virtual-environment/)):

   ```
   $ conda create -n ReID python=3.6
   ```

2. Activate your virtual environment (all the following steps will be done in this activated virtual environment):

   ```
   $ source activate ReID 
   ```

   OR you can use:

   ```
   $ conda activate ReID
   ```

3. Install PyTorch & TorchVision (more details can be found [here](https://pytorch.org/get-started/locally/)):

   ```
   $ conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
   ```

4. Install [Apex](https://github.com/NVIDIA/apex) (It is optional, however, it supports for float16 and thus requires less GPU source ):

   ```
   $ git clone https://github.com/NVIDIA/apex.git
   $ cd apex
   $ pip install -v --no-cache-dir ./
   ```

5. Install scipy:
   ```
   $ conda install -c anaconda scipy
   ```

6. Install matplotlib:
   ```
   $ conda install -c conda-forge matplotlib 
   ```
   
7. Install yaml:
   ```
   $ conda install -c anaconda yaml 
   ```
   
8. After installing dependencies, you can deactivate your virtual environment:

   ```
   $ source deactivate
   ```

   Or you can use:

   ```
   $ conda deactivate
   ```

### 1.3 Get Dataset

* Download the dataset from [here](http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip).


## 2 Scripts/Directories Introduction
This section introduces the scripts and directories in this implement code. The directory structure tree is shown below:
```
.
├── dataset                         /* The directory to hold dataset.                  
├── model                           /* The directory to hold trained model(s).
├── reference                       /* The directory to hold references.
├── preprocessing.py                /* The script to preprocessing dataset.
├── model.py                        /* The core code defines network structures.
├── train.py                        /* The code to train network models.
├── test.py                         /* The code to test network models.
├── evaluate_gpu.py                 /* The code to evaluate network models.
├── demo.py                         /* The demo code to visulization.
```


## 3 Usage 

### 3.1 Preprocessing Data

Extract the dataset and place it into the ```dataset/```. The folder is organized as:

You should now see the ```dataset/``` being organized as:
```
dataset/
├── Market/
│   ├── bounding_box_test/          /* Files for testing (19732 candidate images pool)
│   ├── bounding_box_train/         /* Files for training (12936 images)
│   ├── gt_bbox/                    /* Files for multiple query testing (25259 images)
│   ├── gt_query/                   /* We do not use it 
│   ├── query/                      /* Files for testing (3368 query images)
│   ├── readme.txt                  /* Dataset description
```

Run the following command to process your dataset:

```
$ python preprocessing.py [options]
```
The ```[options]``` here is the dataset directory, of which the default value is ```dataset/Market/```. If you want to change the default value, you can give your own directory while running the command above. It will create a sub-folder called ```pytorch``` under the ```Market``` folder and now the file structure should look like this: 

```
dataset
├── Market/
│   ├── bounding_box_test/          /* Files for testing (candidate images pool)
│   ├── bounding_box_train/         /* Files for training 
│   ├── gt_bbox/                    /* Files for multiple query testing 
│   ├── gt_query/                   /* We do not use it
│   ├── query/                      /* Files for testing (query images)
│   ├── readme.txt
│   ├── pytorch/
│       ├── train/                   /* train 
│           ├── 0002
|           ├── 0007
|           ...
│       ├── val/                     /* val
│       ├── train_all/               /* train+val      
│       ├── query/                   /* query files  
│       ├── gallery/                 /* gallery files  
```

In every sub-dir, such as `pytorch/train/0002`, images with the same ID are arranged in the folder.
Now we have successfully prepared the data for `torchvision` to read the data. For Market-1501, the image name contains the identity label and camera id. Check the naming rule at [here](http://www.liangzheng.org/Project/project_reid.html).



### 3.2 Construct Neural Network Model

There are several pre-trained networks, including [AlexNet](https://medium.com/@smallfishbigsea/a-walk-through-of-alexnet-6cbd137a5637), [VGG16](https://www.kaggle.com/keras/vgg16), [ResNet](https://towardsdatascience.com/introduction-to-resnets-c0a830a288a4), and [DenseNet](https://towardsdatascience.com/review-densenet-image-classification-b6631a8ef803). Those pre-trained networks in general can help us achieve a better performance as they preserve some good visual patterns from [ImageNet](http://image-net.org/index). Fortunately, it is very easy to import them in PyTorch. 

```
from torchvision import models
model = models.resnet50(pretrained=True)
print(model)
```

There are 751 classes (different people) in Market-1501 while there are 1,000 classes in ImageNet. Thus we need to change the model to fit our classifier. The code below is an example to customize the model to 751 classes by passing a parameter named ```class_num```. More details can be found in ```model.py```.

```
import torch
import torch.nn as nn
from torchvision import models

# Define the ResNet50-based Model
class ft_net(nn.Module):
    def __init__(self, class_num = 751):
        super(ft_net, self).__init__()
        #load the model
        model_ft = models.resnet50(pretrained=True) 
        # change avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num) #define our classifier.

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.squeeze(x)
        x = self.classifier(x) #use our classifier.
        return x
```

### 3.3 Training

After having the proper dataset and network model structure, then we can train a model:

```
$ python train.py --gpu_ids 0 --name ft_ResNet50 --train_all --batchsize 32  --data_dir your_data_path
```

Illustrations about arguments:

* ```--gpu_ids``` : which GPU will be used to run the code; You can use ```$ nvidia-smi``` to check your GPU information.

* ```--name```: the directory/path name of the model;

* ```--train_all```: using all images to train the model;

* ```--batchsize```: the training batch size;

* ```--data_dir```: the training dataset directory/path (e.g., here we should set it as ```dataset/Market/pytorch```).

  The training process really depends on your computer or server. For example, it takes me ```19 minutes 28 seconds```  for each epoch.  I use a computer with 16G RAM, Intel Core i7, GPU GeForce GTX 965M 4G. 

### 3.4 Test

After training the model, then it is possible to load the network weight/model to extract the visual feature of every image.

```
$ python test.py --gpu_ids 0 --name ft_ResNet50 --test_dir your_data_path  --batchsize 32 --which_epoch 100
```

Illustrations about arguments:

- ```--gpu_ids``` : which GPU will be used to run the code; You can use ```$ nvidia-smi``` to check your GPU information.
- ```--name```: the directory/path name of the model;
- ```--batchsize```: the training batch size;
- ```--which_epoch```: select the i-th trained model;
- ```--data_dir```: the test dataset directory/path.

### 3.5 Evaluation

After we have features for every test image, then it will be possible for us to match images by using those features. The code below will sort the predicted similarity score.

```
$ python evaluate_gpu.py
```

### 3.6 Visualization

It is possible to visualize the result by using the code below. The ```--query_index``` indicates the image you want to visualize/test. There are 3367 test images in total, thus this number should in the range of [0,3367].

```
python demo.py --query_index 0
```

## 4 Further Reading

1. [Open-ReID](https://cysu.github.io/open-reid/index.html).
2. [(Blog)Person Re-identification](https://amberer.gitlab.io/papers_in_ai/person-reid.html).
3. [A Practical Guide to Person Re-Identification Using AlignedReID](https://medium.com/@niruhan/a-practical-guide-to-person-re-identification-using-alignedreid-7683222da644).

## Reference

1. [(Book) Person Re-Identification](https://link.springer.com/book/10.1007/978-1-4471-6296-4#about).
2. [Person Re-identification: Past, Present and Future](https://arxiv.org/abs/1610.02984).
3. [Spatial-Temporal Person Re-identification](https://arxiv.org/pdf/1812.03282v1.pdf).
4. [Bag of Tricks and A Strong Baseline for Deep Person Re-identification](https://arxiv.org/pdf/1903.07071v3.pdf).
5. [Joint Discriminative and Generative Learning for Person Re-identification](https://arxiv.org/pdf/1904.07223v2.pdf).
6. [Incremental Learning in Person Re-Identification](https://arxiv.org/pdf/1808.06281v5.pdf).


