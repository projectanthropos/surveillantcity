**Implementation/Documentation/review by Taihui Li, RA, assisting the research agenda of Vahan M. Misakyan ©SurveillantCityLab 2018-19**

&nbsp;
&nbsp;


# Host Deep Learning Models on AWS EC2

## Table of Contents
1. [Create AWS Account](#1-create-aws-account)
2. [Launch AWS Instance](#2-launch-aws-instance)
3. [Login and Logout ](#3-login-and-logout)
4. [Upload Your Code and Run](#4-upload-your-code-and-run)
5. [Download Files to Local Machine](#5-download-files-to-local-machine)
6. [Close AWS Instance](#6-close-aws-instance)
7. [Other Information](#7-other-information)



## 1 Create AWS Account

Before moving to try your deep learning mode on AWS EC2, you first should have the account on Amazon Web Services.  Thus, first things first, let's create an AWS account.

1. Visit the [AWS website](https://aws.amazon.com/) and click the button ```Create an AWS Account``` on the top-right corner.

2. You need a valid email, phone number, and a credit card to create your account.

3. After creating your account, there is a page that shows different service plan. I chose the ```basic plan```  which is free. You can choose different plans according to your own requirements. 

   

## 2 Launch AWS Instance

Now you have your own AWS account, you then want to launch a virtual machine (EC2 virtual server instance) on which you can train your deep learning model. Follow the steps below:

1. Log in your AWS account [here](https://aws.amazon.com/).

2. Chose ``` Launch a virtual machine with EC2 ``` .

3. Select ```US West Orgeon```  from the drop-down in the top right hand corner. 

4. There is a list of Amazon Machine Image (AMI). Type ```Deep Learning AMI``` in the search box, then select ```Deep Learning AMI (Ubuntu) Version 24.0```.

5. Then select the hardware to run the AMI. Here I chose ```p3.2xlarge```. This includes a Tesla V100 GPU that we can use to significantly increase the training speed of our models. It also includes 8 CPU Cores, 61GB of RAM and 16GB of GPU RAM.

6. Click ```Review Instance Launch```, you can see the detailed configuration information. 

7.  Click ```Launch```.

8.  Select Your Key Pair.

   * If you have a key pair because you have used EC2 before, select ```Choose an existing key pair``` and choose your key pair from the list. Then check ```I acknowledge…```.

   - If you do not have a key pair, select the option ```Create a new key pair``` and enter a ```Key pair name``` such as ```my_first_aws_keypair```. Click the ```Download Key Pair``` button.

9.  Click ```Launch Instances```. If launch successfully, you will see ```Your instances are now launching```.
10. Click ```View Instances``` to view the status of your instance. The ``` Instance State``` shows ```running```.
11.  Before leaving, there is one thing we need to do. On your own laptop, go to the directory where you just downloaded your key-pair. And then restrict the access permissions on your key pair file. This is required as part of the SSH access to your server.  For example, I used command below: (I download the key-pair under ```~/Downloads/```)

   ```
   $ cd ~/Downloads/
   $ chmod 400 my_first_aws_keypair
   ```



## 3 Login and Logout 

**All the following steps are based on Ubuntu 18.04 system, it will work well for Mac OS. However, if you are using Windows sytem, please check [Connecting to Your Linux Instance from Windows Using PuTTY](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/putty.html?icmpid=docs_ec2_console).**

You have already created your AWS account and launched your server instance. Before we actually deploy our programs on this instance, let's first familar with how to login and logout this instance. 

1. On the ```View Instances``` page, find your ```IPv4 Public IP``` and copy it.

2. On your own laptop, open a Terminal and change directory to where you downloaded your key pair. Login to your server using SSH by using the following command: (Pleae note that I downloaded the key pair under ```~/Downloads/``` and for user name here I used ```ubuntu```, a common user name on AWS can be found [here](https://github.com/taihui/RA_Summer2019/blob/master/4_image_synthesis/4_3_AWS/user_name_aws.png).)

   ```
   $ cd ~/Downloads/
   $ ssh -i my_first_aws_keypair.pem ubuntu@34.219.134.15
   ```

3. After logging in successfully, the instance will list the Python environments. Here, I chose  ```TensorFlow(+Keras2) with Python3 (CUDA 10.0 and Intel MKL-DNN)```  by running the command 

   ```  
   $ source activate tensorflow_p36
   ```
   Wait for a few minutes to let the selected Python environment to set up.

4. You can use command ``` exit```  to logout your server instance.

## 4 Upload Your Code and Run

**All the following steps are based on Ubuntu 18.04 system, it will work well for Mac OS. However, if you are using Windows sytem, please check [Connecting to Your Linux Instance from Windows Using PuTTY](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/putty.html?icmpid=docs_ec2_console).**

After the three steps above, you already know ``` how to create your account```, ```how to configure and launch your server instance```, and ```how to login and logout your server instance```. Now, it's time to upload your code to the server instance and actually run it.

1. Open one terminal, and log into the server instance:

   ```
   $ ssh -i my_first_aws_keypair.pem ubuntu@34.219.134.15
   ```

2. After login,  create a directory to host your code. For example, I created a directory named ```FirstAWS``` by using the command:

   ```
   $ mkdir FirstAWS
   ```

3. Open ```another``` terminal on your own laptop, and use ```scp``` to upload your files to the remote host. My local files location is ```~/Downloads/DCGAN/```. I will upload everything inside this directory onto ```FirstAWS``` which is the directory I just created on the remote server instance.

   ``` 
   $ scp -r -i my_first_aws_keypair.pem ~/Downloads/DCGAN/  ubuntu@34.219.134.15:/home/ubuntu/FirstAWS/
   ```

4. After finishing upload, then go to the first terminal (the one we just used in step 1) where the connection is built. And activate the Python environment we want (here I want ```tensorflow_p36```).

   ```
   $ source activate tensorflow_p36
   ```

5. In the Python environment, go to the directory where you execution code is located and you can finally run it.

   ```
   $ cd ~/FirstAWS/DCGAN/
   $ ls
   $ python test.py
   ```

## 5 Download Files to Local Machine

**All the following steps are based on Ubuntu 18.04 system, it will work well for Mac OS. However, if you are using Windows sytem, please check [Connecting to Your Linux Instance from Windows Using PuTTY](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/putty.html?icmpid=docs_ec2_console).**

We already know how to upload our local files/programs onto remote server instance and run our program. What should we do if we want to get the running results? How can we download it into our local machine? Here assume we want to download everything in ```/home/ubuntu/FirstAWS/DCGAN/``` on the remote server instance to our local machine ```~/Downloads/```. The steps are very similar to ```Upload``` in Section 4.

1. On your own laptop, open a terminal, and use command to download files to your local machine.
   
     ``` 
   $ scp -r -i my_first_aws_keypair.pem  ubuntu@34.219.134.15:/home/ubuntu/FirstAWS/DCGAN  ~/Downloads/
   ```
  
## 6 Close AWS Instance

The more time you use your server instance, the more bill you have to pay. Thus you definitely want to actually close your server instance when you are not using it. Please be advised that the ```logout``` in Section 3 is not ```close server instance```. That means, even you have logged out successfully, your remote server instance is still running. Let's actually close it. Please note that, if you close it, it cannot recover/restart anymore. If you just want it to have a temproal break, you can choose ```Stop``` instead of ```Terminate```. 

1. Log into your AWS account.
2. Click EC2.
3. Click ```Instances``` on the left dashboard.
4. Select the ```Instances``` you want to close.
5. Click ```Actions``` and select ```Instance State```, then you will see the actions you can do. 
6. Choose ``` Terminate``` and confirm it. 
7. You actually close this instance. The ```Instance State``` shows ```terminated```.
8. More detailed information can be found on the [official website](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/terminating-instances.html).

## 7 Other Information

1. **Run scripts as a background process**. This will allow you to close your terminal and turn off your computer while your experiment is running. You can do this by making some minor changes on ```Step 5```  in Section 4. Then the training log will be input into ```script.log```.

   ```
   $ cd ~/FirstAWS/DCGAN/
   $ ls
   $ nohup python test.py >script.log 2>&1 < /dev/null &
   ```

   It will give you a progress id. You can use ```jobs -l``` to see the running status of your progress and use ```kill your-progress-id``` to stop a particular progress. Learn more about nohup, please click [here](https://wiki2.org/en/Nohup).

2.  [10 Command Line Recipes for Deep Learning on Amazon Web Services](https://machinelearningmastery.com/command-line-recipes-deep-learning-amazon-web-services/).
