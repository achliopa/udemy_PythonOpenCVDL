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

## Section 2- NumPy and Image Basics

### Lecture 5 - Introduction to Numpy and Image Section

* Section Goals
	* Understand how to work with basics in NumPy
	* understand how to create arrays
	* slice and index elements from arrays
	* open and display images with numpy

### Lecture 6 - NumPy Arrays

* we `import numpy as np`
* we define a list `mylist = [1,2,3]`
* we cast it to an array `myarray = np.array(mylist)`
* for docs shift+tab
* we can generate an evenly spaced array with `np.arange(0,10)` if we want to add a step size of 2 `np.arange(0,10,2)`
* to create multidimensional array we have many options
* to create a 2d 585 matrix of 0s `np.zeros(shape=(5,5))` it is rowsxcols.
* for 1s we use np.ones() we can omis shapes np.ones((2,4)) is the same
* to create random numbers i first have to seed the rng. `np.random.seed(101)` to seed with 101
* after i seed i can generate random ints and feed them in an array `arr = np.random.randint(0.100,10)` makes an array size 10 with random insts between 0 and 99
* seed is of paramount importance as it leads to generating the same random nums
* to find the max num in an array `arr.max()` to get the location (index) of the max `arr.argmax()` same holds for min
* to get the average val of an array `arr.mean()`
* to reshape arrays: i can get the shape of an array with `arr.shape` for arr its (10,). if i do `arr.reshape(2,5)` i get the array in 2x5 shape... total number of elements must be equal or i get an error
* i make a 10x10 ordered array `mat=np.arange(0,100).reshape(10,10)` to get an element by index
```
row=0
col=1
mat[row,col]
```
* this is called indexing. to gt multiple eleemnts by index its called slicing `mat[0:row,0:col]` start(incl):end(excl):step for everything [:]
* to slice a column `mat[:,1].reshape(10,1)` to slice a row `mat[0,:]`
* to grab a submatrix `mat[:3,:3]`
* to copy an array `mynewmat = mat.copy()`

### Lecture 7 - What is an Image?

* each image can be represented as an array
* in grayscale images the color is represented as a float between 0 and 1 (white = 0 black=1)
* often default images have vals between 0 and 255 8bit resolution
* we can always divide the integer by max val to normalize between 0-1
* what about color images? colorimages can be represented as a comination of Red,Green Blue (additive dcolor mix)
* RGB allows to produce a range of colours (color triangle)
* later in course we will learn about alternative colour mappings that can be applied to images
* each color channel has intensity val 0-255
* when we read a color image with computers or python. the image has 3 dimansions and is a 3d matrix of size (W,H,3) e.g (1280,720,3) 1280 pixel width, 720 pixels height, 3 color channels
* computer does not know about colours. only intensity vals. much like greyscale
* the user has to dictate which channel is for which color.
* each  channel alone is essentially  a grayscale image

### Lecture 8 - Images and NumPy

* we import numpy
* we install matplotlib `conda install matplotlib`
* we import pyplot and make it inline
```
import matplotlib.pyplot as plt
%matplotlib inline
```
* we install pillow `conda install -c anaconda pillow`
* we import Image pillow lib `from PIL import Image`
* Image function allows us to open up images and transform them in an array
* we use it to open an image `pic = Image.open('../DATA/00-puppy.jpg')` 
* if i run `pic` in jupyter i see the pic
* `type(pic)` gives 'PIL.JpegImagePlugin.JpegImageFile'  . munpy cant process it. to convert it to an array i use `pic_arr = np.asarray(pic)`
* `pic_arr.shape` gives (1300, 1950, 3). i can show the image from the array `plt.imshow(pic_arr)`
* i can show first channel as grayscale `plt.imshow(pic_arr[:,:,0],cmap='gray')`
```
pic_red = pic_arr.copy()
pic_red[:,:,0]
```
* the red channel by default has a viridis colormap becaus the vals are 0-255. we have to normalize by dividing by 255 to have 0-1 scale (greyscale) [colorscales](https://matplotlib.org/examples/color/colormaps_reference.html)
* lighter color in grayscale is closer to 255 (or 1.) so higher color contribution in pixel
* i will zero out green and blue channel channel `pic_red[:,:,1:] =0` and show it `plt.imshow(pic_red)`
* pic red has still 3 channels they are just zeroed out

### Lecture 9 - Numpy and Image Assesment Test

* we do the test
* fill an empty array with vals
```
arr=np.empty(shape=(5,5))
arr.fill(10)
```
* or
```
arr,np.ones((5,5))
arr*10
```

## Section 3 - Image basics with OpenCV

### Lecture 11 - Introduction to images and OpenCV Basics

* we will learn how to use OpenCV lib
* how to open images and draw on them
* OpenCV (Open Source Computer Vision) is a library of programming functions mainly aimed at real-time computer vision
* Created by Intel at 1999, is written in C++. Here we will use its Python bindings
* It contains many popular algorithms for computer vision, including object detection and tracking algorithms
* Section Goals
	* open inage files with OpenCV in a notebook and in py script
	* Draw simple geometries on images
	* Direclty interact with an image through callbacks

### Lecture 12 - Opening Image Files in a notebook

* we asaw how to use the PIL(PythonImagingLibrary) to open images and trasform them to arrays with numpy and use matplotlib to display the array as image
* we will use OpenCV + Matplotlib to open and display an image as array
* we do the usual imports
```
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```
* we import openCV lib `import cv2`
* we open the image with cv `img = cv2.imread('../DATA/00-puppy.jpg')` the type is `type(img)` is numpy.ndarray
* if i open wrong path i get no error but he type is NoneType
* the `img.shape` is (1300,1950,3) so 3 channels
* if i `plt.imshow(img)` the image is of diff color because in OpenCV color channels have differt order
* Matplotlib expects RED, GREEN, BLUE but openCV encodes them BLUE,GREEN,RED
* we need to fix the order before displaying with matplotlib. cv can do that using the cvtColor function `cv2.cvtColor(img,cv2.COLOR_BGR2RGB)` cv2 has a lot of colorplane transformations available
* i can avoid the post transofrmation . i can apply it when i read the image with opencv e.g to show it as grayscale `img_gray = cv2.imread('../DATA/00-puppy.jpg',cv2.IMREAD_GRAYSCALE)` if we plot it we see the viridis cmap as the vals are integers (even if i normalize them its stil viridis) to solve it i change cmap `plt.imshow(img_gray,cmap='gray')`
* to resize images we can use openCV `new_img = cv2.resize(fixed_img,(1000,400))` th enumbers i enter are (COl,ROW) or (WIDTH,HEIGHT) if i dont keep aspect ratio the image is transformed. the arguments are swapped in comparizon with numpy order
* cv allows resizing keeping the aspect ratio
```
w_ratio = 0.1
h_ratio = 0.1
new2_img = cv2.resize(fixed_img,(0,0),fixed_img,w_ratio,h_ratio)
```
* to flip images `fl_img = cv2.flip(fixed_img,0)` along the horizontal axis, use 1 to flipalong the horizontal. use -1 to combine both flips
* to write an image (nupy array) maybe a generated one to anew file i use `cv2.imwrite('NEW FILEPATH',fl_img)` the filetype code i use determines the filetype. beware that as OpenCV does the save it saves them in BGR order 
* to play with canopy space in notebook to display larger images we do matplotlib scripting
```
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
ax.imshow(fix_img)
```