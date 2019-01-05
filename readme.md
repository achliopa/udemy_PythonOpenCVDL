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

### Lecture 13 - Opening Image files with OpenCV

* we will open images with OpenCV using python scripts.
* in this lecture we will use OpenCV to display images in their own separate window outside of jupyter
* for more complex video and image analysis, we will need to display outside of jupyter
* while we often will just use plt.imshow() to display images inside of a notebook. sometimes we want to use OpenCV on its own to display images in their own window
* Often Jupyter (being browser based) interferes with closing teh window
* Many times  JupyterLab can display a new window with no issues, but the kernel crashes when the OpenCV window is closed
* To fix this issue if running OpenCV from notebook. restart the kernel
* this is an issue of MAcOS and linux
* it better to run the code in a .py script if the issue makes our work difficult.
* we ll see how to open and display images direclty with OpenCV (no matplotlib) in the notebook and in a script
* we start with a notebook
* we do some fixes first to run opencv window in lunux
```
Being in the course's conda env source activate python-cvcourse

conda remove opencv
conda remove py-opencv
conda update conda
conda upgrade pip
conda install jupyter # dont think it matters but followed instructors advice
then use pip to install opencv pip install opencv-contrib-python (i tried this version as it contains additional libs, I suppose pip install opencv-python will also do the trick)
```
* then we use imshow from opencv to invoke the window 
```
import cv2
img = cv2.imread('../DATA/00-puppy.jpg')
cv2.imshow('Puppy',img)
cv2.waitKey()
```
* the image is large and we cannot resize it as opencv displays on same pixel dimensions. so depending on teh screen  analysis this  might cause issues
* we write a python script to do the same job
```
import cv2
img = cv2.imread('../DATA/00-puppy.jpg')
while True:
    cv2.imshow('Puppy',img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cv2.destroyAllWindows()
```
we puth th imshow in a while loop to be able to brake on keystroke we use the cryptic `cv2.waitKey(1) & 0xFF == 27:` that means IF we ve waited atleast 1ms and weve pressed the ESC key
* instead of 27 (ESC) we can use `ord('q')` to quit with 'q'

### Lecture 14 - Drawing on Images - Part One - Basic Shapes

* we make a new notebook and do the basic imports
```
import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```
* we create a blank image (all zeroes) in numpy `bblank_img = np.zeros(shape=(512,512,3),dtype=np.int16)` we spec the datatype to be int16
* we imshow it and its pureblack
* we will use opencv to draw on the image a rectangle `cv2.rectangle(blank_img,pt1=(384,0),pt2=(510,150),color=(0,255,0),thickness=10)` the code draws a rectangle specing 2 vertexes with two opposite edge points. also we spec the color and the thickness . the definition of the points is in OpenCV style W,H. the method alters the passed image so if we replot it with imshow we see the overlay rect
* outline starts at the specked points so in our example it goes out of bounds
* if i run multiple times the method it overlays multiple rects
* the same holds for squares
* for circles `cv2.circle(img=blank_img,center=(100,100),radius=50,color=(255,0,0),thickness=8)`
* to fill a shape witht he color we set thickness to -1
* to draw a line `cv2.line(blank_img,pt1=(0,0),pt2=(512,512),color=(0,255,255),thickness=5)`

### Lecture 15 - Drawing on Images - Part 2 - Texts and Polygons

* to write text
```
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(blank_img,text="Hello",org=(10,500),fontFace=font,fontScale=4,color=(255,255,255),thickness=3,lineType=cv2.LINE_AA)
```
* we selecta cv2 font, the bottom left corner, the size, color,thickness and linetype
* to draw a polygon with opencv we have first to decide on the vertices alist of pairs as nested arrays with x,y coordinates. the dtype has to be the same as the image `vertices = np.array([[100,300], [200,200], [400,300], [200,400]],dtype=np.int32)`
* the shape of the vertices is (4,2) so 2D. opencv wants it 3D
* the conversion we do is `pts = vertices.reshape((-1,1,2))` and the `pts.shape` is (4,1,2)
* we do this for the color channels
* to draw the polyline `cv2.polylines(blank_img,[pts],isClosed=True,color=(255,0,0),thickness=5)` we pass the points as array, also we spec if we want to close the polyline

### Lecture 16 - Direct Drawing on an Image with a Mouse - Part One

* we can use CallBacks to connect Images to event functions with OpenCV
* this allows us to directly interact with images (and later on videos)
* In this 2 part lecture we will cover
	* Conecting Callback Functions
	* Adding Functionality through Event Choices
	* Dragging the Mouse for Functionality
* we will run them as python script
* we import libs cv2 and numpy
* we create a blank image `img = np.zeros((512,512,3),np.int8)`
* int8 results in grayish color
* we add the while loop
```
while True:
    cv2.imshow('Blank',img)
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()
```
* we define a callbakc function
```
def draw_circle(event,x,y,flags,param):
    pass
```
* and we connect it witha Mouseevent to the image 
```
cv2.namedWindow(winname='Blank')
cv2.setMouseCallback('Blank',draw_circle)

```
* the conenction is done on the imShow name (window name)
* the params  passed in the callback have to do with the event
	* x,y is the position
	* event contains the type of event
* i mod the callback to draw a circle of specific  size and color centered at the position i click using the event 'cv2.EVENT_LBUTTONDOWN'
```
def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img,(x,y),100,(0,255,0),-1)
```
* we remove np.int8 to solve the grayish look

### Lecture 17 - Direct Drawing on an Image with a Mouse - Part Two

* we mod the callback adding an elif to listen to a RButton down event drawing a circleof another color (red) beware that it is BLUE,RED,GREEN
```
    elif event == cv2.EVENT_RBUTTONDOWN:
        cv2.circle(img,(x,y),100,(0,0,255),-1)
```

### Lecture 18 - Direct Drawing on an Image with a Mouse - Part Three

* we will build the rectangle as we grag the mouse with button down on a blank img and set it on button rais
* we cp the prev script (show image + function boilerplate)
* the callbak becomes
```
################
## VARIABLES ###
################

# true while mouse button down false while up
drawing = False
# starting point of rect temp vals
ix,iy = -1,-1

###############
## FUNCTION ###
###############

def draw_rectangle(event,x,y,flags,param):
    global ix,iy,drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing=True
        ix,iy = x,y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing=False
        cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
```
* we make use of global vars in the callback so we define them as such to be able to alter them

### Lecture 19 - Image basics Assessment

* to fill polyline we can use a beta function `cv2.fillPoly(fix_img,[pts],(0,0,250))`

## Section 4 - Image Processing

### Lecture 21 - Introduction to Image Processing

* Section Goals
	* Learn various image processing operations
	* Perform image operations such as Smoothing,Blurring,Morphological Operations
	* Grab properties such as color spaces and histograms

### Lecture 22 - Color Mappings

* so far we ve only worked with RGB color spaces, in RGB coding, colors are modeled as a combination or Red,Green and Blue
* in the 1970s HSL (hue,saturation,lightness) and HSV (hue,saturation,value) were developed as alternative color models
* HSV and HSL are more closely aligned with the way human vision actually perceives color
* while in the course we will deal more with RGB images, it goog to know how to convert to HSL and HSV colorspaces
* RGB colorspace represents a color as a combo of R G and B (color cube)
* HSL is perceived as cylinder (hue is the angle, saturation is the distance from center, lightness the height)
	* H= actual color, Saturation=Intensity of color, Lightness=How dark it is
	* bottom pure black, top pure wight, center line = grayscale
* HSV is represented as cylinder. instead of lightness we have value (black->full color)
	* top center = white
* this lecture will be a quick review on using the cvtColor func to change colorspaces
* we wont have to deal with HSL or HSV based color images for the rest of the course
* we use a notebook to display an image at default cv2 BGR colorspace
```
import cv2
import matplotlib.pyplot as plt
%matplotlib inline
img = cv2.imread('../DATA/00-puppy.jpg')
plt.imshow(img)
```
* we fix it converting to RGB
```
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img)
```
* to convert it at HSV `img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)` the image is strange because color channels have RGB vals and are displayed as HSV. we can use cv2.COLOR_RGB2HLS for HLS

### Lecture 23 - Blending and pasting Images

* often we will work with multiple images
* OpenCV has many programmatic methods of blending images together and pasting images on top of each other
* Blending Images is done through the **addWeighted** function that uses both images and combines them
* To blend images we use a simple formula:
	* `new_pixel=α*pixel_1+β*pixel_2+γ`
* so it adds weights to each contributing image pixel and adds a bias
* when images are not same size we have to do masking
* we read 2 images and fix the color order for display
```
img1 = cv2.imread('../DATA/dog_backpack.jpg')
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
img2 = cv2.imread('../DATA/watermark_no_copy.png')
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
```
* we import matplotlib and show them. they are not the same shape
* well resize them both to make them same size
```
img1 = cv2.resize(img1,(1200,1200))
img2 = cv2.resize(img2,(1200,1200))
```
* we use the addWeighted function to blend them `blended = cv2.addWeighted(src1=img1,alpha=1,src2=img2,beta=0.3,gamma=0.5)` its src1, alpha,src2,beta,gamma
* if we blend different size images we get an error
* we will overlay a small image on top of a larget image without blending. 
* its a simple numpy reassignemnt where the vals of the larger image will be reassigned to equal the vals of the smaller image on the overlayed space
* we resize img2 to be smaller `img2 = cv2.resize(img2,(600,600))`
* we rename images `large_img = img1` and `small_img = img2`
* overlay is pure numpyarray math
```
x_offset = 0
y_offset = 0
x_end = x_offset + small_img.shape[1]
y_end = y_offset + small_img.shape[0]
large_img[y_offset:y_end,x_offset:x_end] = small_img
```

### Lecture 24 - Blending and Pasting Images Part Two - Masks

* we ve seen how to overlay images on top of each other by simply replacing values of the larger images with vals of the smaller image for the desired RegionOfInterest
* what if we only want to blend or replace part of the image?
* what if we want to mask part of the smaller image. say replace only the area in the outline of a logo
* this needs 3 steps. start with  img1 => build a mask(the mask will let only certain pixels of img1 filter through) => paste the masked pixels on img2
* lets explore the sysntax of these steps (check links in lecture notebook for other use cases)
* we start witht he same 2 images (read and fix and resize logo img)
* we decide where on the base img(img1) we want to blend in the img2 shape (create a ROI). we ll place it in bottom right (numpy array math ahead)
```
x_offset = img1.shape[1]-img2.shape[1]
y_offset = img1.shape[0]-img2.shape[0]
```
* what these vals represent is the topleft corner of ROI coorsinates
* i use tuple unpacking to get img2 dimensions `rows,cols,channels = img2.shape`
* i grab the ROI `roi = img1[y_offset:img1.shape[0],x_offset:img1.shape[1]]`
* i now want to create the mask
	* i get a grayscale version of the image `img2gray = cv2.cvtColor(img2,cv2.COLOR_RGB2GRAY)` its in viridis cmap but is 1 channel
	* i need to inverse the image because i want with black(0 val) the part to be excluded. we use cv2.bitwise_not (bitwise inversion) for this `mask_inv= cv2.bitwise_not(img2gray)`
* `mask_inv.shape` shows is 1 channel i need to add the other channels (with numpy)
* we create a white 3channel background for the size of img2 (mask) `white_background = np.full(img2.shape,255,dtype=np.uint8)` full numpy method fills a speced sized array (shape) with the number we spec (255). as it fills 255 in all cahnnels its white. also we spec dtype=np.uint8 to match the mask dtype
* to create the actual mask we use cv2.bitwise_or (bitwise disjunction per element)  `bk = cv2.bitwise_or(white_background,white_background,mask=mask_inv)` the result has 3 channels but is essentialy the mask. it applies the mask in all channels. we could just cp the 1 channel in others with numpy
* we now want to apply the original im2 (red) on the mask to cut the logo out and create the foreground (we use again bitwise_or) `fg = cv2.bitwise_or(img2,img2,mask=mask_inv)`
* we get the mask overlayed on the roi with biwise_or (not masked) `final_roi = cv2.bitwise_or(roi,fg)`
* we use overlay to overlay roi on the original large image (like we dit before with numpymath)
```
large_img = img1
small_img = final_roi
x_end = x_offset + cols
y_end = y_offset + rows
large_img[y_offset:y_end,x_offset:x_end] = small_img
```

### Lecture 25 - Image Thresholding

* In some CV applications it is often necessary to convert color images to grayscale, since only edges and up being important
* Similarly, some apps only require a binary image showing only general shapes
* Thresholding is fundamentally a very simple method of segmenting an image into different parts
* THresholding will convert an image to consist of only two values, white or  black
* what we actually do is convert a color image to binary (3channels -> 1channel, unit8 => binary)
* We ll dive in syntax anoptions for thresholding with OpenCV
* we do usual imports and read an image of a rainbow `img = cv2.imread('../DATA/rainbow.jpg')`
* we ll see some thresholding options
* read in a color image as grayscale `img = cv2.imread('../DATA/rainbow.jpg',0)` simply pass a 0
* use **cv2.thresholding** passing options thresh and maxval and type of threshold. so any val <thresh is converted to 0 each val >thresh to maxval. usually we use the halfway point. foa grayscale image th typical is `ret,thresh1 = cv2.threshold(img,t27,255,cv2.THRESH_BINARY)` ret is the cutoff value and thresh1 is thresh1 is the actual image thresholded
* we can play with threshold types like THRESH_BINARY_INV (inverse) THRESH_TRUNC (if val is over threshold it replaces it with threshold, if its lower it keeps the original val) THRESH_TOZERO (keep original if >thresh otherwise 0)  (see OpenCV docs for more)
* we will do a real world example reading in a crossword page image `img = cv2.imread('../DATA/crossword.jpg',0)` 
* we set afunction to display pyplot larger and use it insteat of plt.imshow
```
def show_pic(img):
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap='gray')
```
* we see in the image that apart from black letters there is gray noise. we wold like to say: if there is ink its black if not white. we ll play with binary threshold
* we do simple binary in middle `ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)`. the result is not perfect as we loose quality. we can play with types or level. level is not very helpful. THRESH_OTSU and TRIANGLE do a good job
* a better approach is the adaptive trheshold as it auto adapts the threshold based on pixel and neighboring pixels `th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,8)` its params:
	* srcimage
	* maxval
	* adaptive threshold type (algorithm) GAUSSIAN or MEAN
	* threshold type (the actual threshold)
	* neghbour size for adaptibe threh algo (texel) only odd nums
	* the C val to be subtr. from sum (see docs)
* we usually play with block size and c val (2 last params)
* we can now start apply multiple methods like blending adaptive thresholded image with binary thresholded to see the result `blended = cv2.addWeighted(th1,0.5,th2,0.5,0)`

### Lecture 26 - Blurring and Smoothing

* a common operation for image proc is blurring and smoothing an image
* smoothing an image can help get rid of noise, and help the app focus on general details
* there are mnay methods for blurring and smoothing
* often blurring and smoothing is combined with edge detection
* edge detection algos show many edges when shown a high res image with no blurring
* edge detection after blurring gives better results
* Blurring Methods we ll explore:
	* Gamma Correction: gamma correction can be applied to an image to make it appear brighter or darker depending on the Gamma value chosen
	* Kernel Based Filters: Kernels can be applied over an image to produce a variety of effects. To understand what is check [Interactive visualization](http://setosa.io/ev/image-kernels/) the examples apply a 3x3 kernel of predef vals dependign the effect we want rolling over the image to apply the effect. we see the original pixels as matrix multiplied with filter kernel vals and the resulting pixel. also in borders ixels are unknown

### Lecture 27 - Blurring and Smoothing - Part Two

* [Tutorial](https://www.tutorialspoint.com/dip/concept_of_blurring.htm)
* we open a notebook and do the normal imports
* we add convenience read img func
```
def load_img(name):
    img = cv2.imread(name).astype(np.float32)/255
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img
```
* we read an img `img = load_img('../DATA/bricks.jpg')`
* the image is in float forma of 0-1 scale
* we add a display image helper func
```
def display_img(img):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img)
```
* we start with gamma correction. we define a gamma <1 and raise the image numpy array to the power of gamma. effectively making the image faded. if gamma >1 the image is more intense(darker)
```
gamma = 1/4
result = np.power(img,gamma)
```
* we do our first blurring. a low pass filter witha 2d convolution
* we will write on the image to show the effect
```
img = load_img()
font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img,text='bricks',org=(10,600),fontFace=font,fontScale=10,color=(255,0,0),thickness=4)
display_img(img)
```
* this font is conventient becaus the contour of letters are clear lines. the space betweenlines in letters will be affected by blurring or smoothing
* we setup the kernel for the filter our kernel is 5x5 of 1/25 (0.04) val `kernel= np.ones(shape=(5,5),dtype=np.float32)/25`
* we apply a 2d filter on it with cv2.filter2D. `dst = cv2.filter2D(img,-1,kernel)` we pass in
	* input image
	* desired depth (ddepth) of the destination image. if using -1 its the same as input image
	* the kernel
* the result is a blurred image. lines ar emore thick in letters and detail is lost from wall. lines flood
* we reset the image (w/letters) to try a new method (Smooth image with Averaging). its the same as we did before with our custom kernel.. its just we use cv2 built in cv2.blur() method passing the kernel dimensions. the val is the 1/elements in kernel. so it does averaging. just what we did manually before `blurred = cv2.blur(img, ksize=(5,5))`
* increasing kernel size makes the effect more intense
* we will apply Gaussian and Median Blurring (not Averaging)
* Gaussian Blur: `blurred_img = cv2.GaussianBlur(img,(5,5),10)` (src,ksize,sigmavalue)
* Median Blur: `blurred_img = cv2.medianBlur(img,5)` (src,ksizedim) it takes an int as ksize as the kernel is square. this blur is different as lines dont flood as before (it  does remove noise keeping details)
* we do a real example. we imread sammy.jpg (tutors dog) and display it after color correcting
* we imread a noisy version of the picture sammy_noise.jps which is color corrected
* we ll try to fix the noise with median_blur `fixed_img = cv2.medianBlur(noise_img,5)` it works wonders
* we will do bilateral filtering to the brick image 'cv2.bilateralFilter' params: src,d,sigmaColor,sigmaSpace `blur = cv2.bilateralFilter(img,9,75,75)` it blurs keeping edges

### Lecture 28 - Morphological Operators

* [Math](https://en.wikipedia.org/wiki/Mathematical_morphology)
* [Morph Operators](https://homepages.inf.ed.ac.uk/rbf/HIPR2/morops.htm)
* Morphological Operators are sets of Kernels that can achieve a variety of effects such as reducing noise
* Certain operators are very good at reducting black points on a white background (or vice versa)
* Certain operators can also achieve an erosion and dilation effect that can add or erode from an existing image
* this effect is most easily seen on text data, so we will practice various morphological operators on some simple white text on a black background
* we create anotebook and do the normal imports
* we add a helper func to add white text on blabk background
```
def load_img():
    blank_img=np.zeros((600,600))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blank_img,text='ABCDE',org=(50,300),fontFace=font,fontScale=5,color=(255,255,255),thickness=25)
    return blank_img
```
* we add a func to deiplay the img
```
def display_img(img):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap='gray')
```
* we start with errosion (it errodes boundaries of foreround objects) (detect edges and erode boundary)
	* we define a 5,5 ones kernel `kernel=np.ones((5,5),dtype=np.uint8)`
	* we apply 'cv2.erode' specing src,kernel,iterations `result = cv2.erode(img,kernel,iterations=1)` the more iterations the more eroded the boundary of the letter (finer) over 5 we lose the letters
* we follow with opening (erosion followed with dilation). opening removes background noise
	* we add some binary whitenoise as overlay to original image

	```
	img = load_img()
	white_noise = np.random.randint(low=0,high=2,size=(600,600))
	white_noise = white_noise*255
	noise img = white_noise + img
	display_img(noise_img)
	```
	* we do opening using the cv2.morphologicalEx `opening = cv2.morphologyEx(noise_img,cv2.MORPH_OPEN,kernel)` using the same 5x5 kernel we use for erode. the result is a noise image. boundary is not 100% perfect but is very very good
* sometimes we have foreground noise. we use then Closing to clean
	* we create black noise on the image (is like white noise but reverse as we multiply by -255). it wonth affect the black background but will affect the white foreground
	* black noise subtyracts 255 to random pixels making it darker, white noise adds 255 to random pixels

	```
	img = load_img()
	black_noise = np.random.randint(low=0,high=2,size=(600,600))
	black_noise = black_noise * -255
	black_noise_img[black_noise_img== -255] = 0
	display_img(black_noise_img)
	```
	* we apply closing. (like opening but with other morph operation) `closing = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)` result is OK
* Morphological gradient takes the difference between dilation and erosion of an image
	* (errosion will eat the foregrounf making it thinner, dialtion will thicken the foreground)
	* morph gradinet will take the difference of two. what we get is the egde or contour of the foreground shape. thsi is a way of edge detection `gradient = cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)` it does a pretty good job

### Lecture 29 - Gradients

* [Image Gradients](https://en.wikipedia.org/wiki/Image_gradient)
* [Sobel Operator](https://en.wikipedia.org/wiki/Sobel_operator)
* understanding gradents will lead to understand edge detection which applies to object detection, tracking and image classification
* an image gradient is a directional change in the intensity of color in an image. there are algos that can track this direction
* in this lecture we will mainly explore basic Sobel-Feldman operators 
* later in course we will expand on this operator for general edge detection
* gradients can be calculated in a specific direction
* if we use a normalized-x gradient from Sobel operator we see edges mainly on the vertical axes
* if we use a normalized-y gradient from Sobel operator we see edges mainly on the horizontal axes
* a normalized gradient magnitud from Sobel operator detects edges on both axes
* the operator uses two 3x3 kernels which are convoluted with the original image to calculate approximations of the derivatives. one for the horizontal changes and one for vertical
	* Gx = [[+1,0,-1],[+2,0,-2],[+1,0,-1]] * A
	* Gy = [[+1,+2,+1],[0,0,0],[-1,-2,-1]] * A
* We ll expolre various gradient operators with OpenCV
* We ll also combine these concepts with a few other image processing techniques we ve learned
* we open a notebook, do the imports and add the display image helper
* we read in a sudoku img in grayscale `img = cv2.imread('../DATA/sudoku.jpg',0)` it has vertical and horizontla lines and nums
* we apply sobel on the x direction `sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)`. we use cv2.Sobel func with params:
	* source image,
	* ddepth (desired depth) selcting from  OpenCV available depths it has to do with the desired level of detail
	* x derivative (1 as we want to apply in that direction)
	* y derivcative (0 as we ignore that direction)
	* ksize=5 (sqaue kernel only 1 direction)
* the result is as expected.detects vertical edges
* we apply sobel on y direction `sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)` it detects horizontal lines
* another gradient uses laplacian derivatives. we can calculate these using sobel operators `laplacian = cv2.Laplacian(img,cv2.CV_64F)` we use cv2.Laplacian passing src image and ddepth. it does a good job in bothj directions
* a use case of this iage could be to do edge detection to detenc numbers in the image
* we might want the combined result of sobelx and sobely. we can use addWeighted `blended = cv2.addWeighted(src1=sobelx,alpha=0.5,src2=sobely,beta=0.5,gamma=0)`, 
* a second step in the pipeline could be to do thresholding or apply morphological operators `ret, th1 = cv2.threshold(blended,100,255,cv2.THRESH_BINARY_INV)` then later do openning to remove noise and so on. or apply morphological gradient `gradient = cv2.morphologyEx(blended,cv2.MORPH_GRADIENT,kernel)`

### Lecture 30 - Histograms - Part One

* We ll understand what a regular histogram is, then we ll explain what an image histogram means
* A histogram is a visual representation of the  distribution of a continuous feature
* its a typical plot in data analysis (pyplot offes it seaborn as well), usually we specify a set of bins and display the frequency of a number being in the bin as a barchart
* we can display it as a genral trend of the frequency drawing a like (KDE plot)
* for images we can display the frequency of values for colors
* each of the three RGB channels has vals between 0-255
* we can plot these as 3 histograms on top of each other to see how much of each channel there is in the picture
* we ll see how to create picture histograms with matplotmlib and OpenCV
* we create a notebook and do the useual imports
* we imread 3 images fixing the color fot matplotlib (horse.jpg , rainbow.jpg, bricks.jpg). we keep 2 copies one for show in RGB and one for processing in BGR (openCV)
* horse image has a lot of black so we expect peak near 0 for all channels
* rainbow has even distribution
* in bricks we expect a peak for blue
* to calculate the histogram values we use cv2.calcHist() method `hist_values = cv2.calcHist([blue_bricks],channels=[0],mask=None,histSize=[256],ranges=[0,256])` that takes a s arguments:
	* [sourse_image in BGR openCV format]
	* channel to show 0=b,1=g,2=r
	* mask: if we want to show histogram for a masked part of image (None=show all)
	* histsize is the num of vals (like buckets) of histogram
	* ranges : the range of vals 
* we use plt.plot(hist_values) to plot the histogram in matplotlib
* for blue_bricks the B hist ihas a peak in midle. for dark_horse a peak in 0 as image has no blue
* to plot the 2 color histogram all at once in matplotlib we use for loop and vanilla python
```
img = blue_bricks
color = ('b','g','r')
for i,col in enumerate(color):
    histr=cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color=col)
    plt.xlim([0,256])
plt.title('Histogram for Image')
```
* for dark_horse the histogram is biased as it is a very large picture of mostly pure black so we need to play with plotlimits to see what happens with colors

### Lecture 31 - Histograms - Part Two - Histogram on Masked Portion

* We continue our discussion on histograms with 2 more topics
	* Histograms on a masked portion of the image
	* Histogram Equalization
* As mentioned in the previous lecture we can select a ROI and only calculate the color histogram of that masked section
* we ll see how to create amask to achieve this effect
* histogram equalization is a method of contrast adjustment based on teh images histogram. we saw how we can use gamma correction to increase or degreace the brightness of an image. we will see how to increase or decrease the contrast of an image with histogram equalization
	* we take an image segment (ROI) of a grayscale image and plot its color histogram. the histogram has no vals close to 0 and 255
	* applying histogram equalization will reduce the color depth (shades of gray or inbetween colors)
	* min and max vals in this ROI are 52 and 154. after applying the histogram equalization min is 0 and max is 255 so in essence we increase the contrast
	* histogram is now more evenly distributed or flatened out, high peaks are gone
	* we also see less shaades of gray
	* histogram equalization uses the accumulative histogram. after histogram equalization the accumulative histogram is a linear line from min to max
	* histogram itself maintains the contour but is opened or flatened out
* we ll do both techniques in opencv
* we start by building a mask to cut a ROI in the rainbow image. the mask will be white rectangle on black background. we will use bitwise operation (and) on original
```
rainbow = cv2.imread('../DATA/rainbow.jpg')
show_rainbow = cv2.cvtColor(rainbow,cv2.COLOR_BGR2RGB)
mask = np.zeros(img.shape[:2],np.uint8)
mask[300:400,100:400] = 255
masked_img = cv2.bitwise_and(img,img,mask=mask)
```
* we also get a show version of the masked img to visualy confirm the histogram results
* getiung the masked histogram is easy as we apply the mask in calcHist function. with `hist_mask_values_red = cv2.calcHist([rainbow],channels=[2],mask=mask,histSize=[256],ranges=[0,256])` we get the red hist of the ROI
* to compare we get ared hist for the complete image anfd plot both

### Lecture 32 - Histograms - Part Three - Histogram Equalization

* we load a gorilla image in grayscale and display it using a helper method
* its a large image
* we will visualize the histogram then equalize it and see the difference, then convert it back to color image
* as we work in grayscale we have only one colorchannel to hist `hist_values = cv2.calcHist([gorilla],channels=[0],mask=None,histSize=[256],ranges=[0,256])` we have no pure black colors, and white comes from background
* to equalize histogram we use cv2.equalizeHist() method passing the image `eq_goriila = cv2.equalizeHist(gorilla)` we display it iand it has high contrast. also we get the histogram and plot it
* histogram is flatened out and there are alot of 0s in order to get the linear cumulative hist
* we can apply the equalizeHist in grayscale and color images. for color images we need to convert them to HSV colorspace and use only value channel  in equalization
```
hsv_gorilla = cv2.cvtColor(color_gorilla,cv2.COLOR_BGR2HSV)
value_channel = hsv_gorilla[:,:,2]
eq_value_channel = cv2.equalizeHist(value_channel)
hsv_gorilla[:,:,2] = eq_value_channel
show_eq_color_gorilla = cv2.cvtColor(hsv_gorilla,cv2.COLOR_HSV2RGB)
display_img(show_eq_color_gorilla)
```
* the histogram plot of the value channel is the same as of the grayscale version of the image

### Lecture 33 - Image Processing Assesment

* 