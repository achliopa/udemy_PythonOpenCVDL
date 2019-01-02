# Udemy Course: Python for Computer Vision with OpenCV and Deep Learning

* [Course Link](https://www.udemy.com/python-for-computer-vision-with-opencv-and-deep-learning/)
* [Course Repo]()

## Section 1 - Course Overview and Introduction

### Lecture 3 - Course Curriculum Overview

* Goals of this COurse
	* Understand Computer Vision Apps
	* Understand how to use OpenCV and Python work with Images and Vectors
	* Be able to apply these skills in our projects
* Numpy and Image basics
	* Quick section on NumPy basics and how to manipulate images with it
* Image Basics with OpenCV
	* Begin to  work with the OpenCV library  with images
	* Basic commands and drawings on Images
* Image Processing with OpenCV
	* understand more advanced OpenCV operations that are useful in real world apps
* Video Processing with OpenCV
	* understand the basics of working with video files and streaming webcam video with OpenCV library
* Object Detection
	* Learn the various different methods of detecting objects in images and videos
	* Start with basic template matching and work our way up to face detection
* Object Tracking
	* expand from our knowledge of object detection to tracking objects in videos
* Deep Learning with  Computer Vision
	* begin to combine knowledge from prev section with latest tools in Keras and Tensorflow for state of the art deep learning apps

### Lecture 4 - Getting Set-Up for the Course Content

* We need to download and install Anaconda
* Create a Virtual Environment (different files depending on OS)
* open JupyterLab
* Work with both notebooks and .py scripts in jupyterlab
* we have a plotly conda env that might serve us (has tensorflow) we use it with `source activate plotly`
* we lanunch navigator with `anaconda-navigator` we also update all with `conda update --all`
* We will list the installation workflow to see if there is sthing new
* get anaconda from 'https://www.anaconda.com/download/' get for python3.7 x64
* when install  if conda is already installed choose to add the PATH
* the tutor gives a .yml file that will create and activate avirtual env for us. so we should do it and maybe delete our own. his has all the necessary libs 
* i will delete my plotly env as the one from course is huge and overlaps it
* i update conda `conda update -n root conda` 
* i remove plotly `conda env remove -n plotly` probably i can remove ztdl also...
* we cd to our course folder and create the env from  the yaml `conda env create -f cvcourse_linux_new.yml`
* to activate env `source activate python-cvcourse` to deactivate `source deactivate`
* to start jupyter-lab we run `jupyter-lab` and it runs as a webAPP at 'http://localhost:8888/lab'
* we create a notebook and a textfile as py. we run the python script int erminal with `python3 myCode/test.py`