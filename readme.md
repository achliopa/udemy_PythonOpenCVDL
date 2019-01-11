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

## Section 5 - Video Basics with Python and OpenCV

### Lecture 35 - Introduction to Video basics

* Goals of this Section
	* Connect OpenCV to a WebCam
	* Use OpenCV to open a video file
	* Draw Shapes on video
	* interact with video

### Lecture 36 - Connecting to Camera

* we ll see how to connect with openCV to a usb camera on the laptop or the built-in camera of the laptop
* also we will see how to video stream from the camera to a file using openCV
* when we read video data is important to not have multiple notebooks or files running
* this will create conflicts to openCV
* we should have only one file reading from camera. the others should have their kernels shutdown
* a running notebook has a green dot in file tree of jupyter
* to connect to a camera
	* import cv2
	* create a capture object with cv2.VideoCapture passing the index of the input device we will use `cap = cv2.VideoCapture(0)`, for us 0 is the built in camera and 1 is a usb webcam of better quality
	* we grab the height and width of the frame to use it in processing (they are floats so we cast them to int)
	
	```
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	```
	* display the imaage
	
	```
	while True:
    	ret,frame = cap.read()
	```
* what 'cap' is actually is a series of images. a stream of images. 
a frame is a single image
* a video is a contiusly updated frame
* we apply the methods we learned for images on frames. to get the current frame we use cap.read() continuously
* our processing happens in the while loop (we can add escape logic like before)
* to convert the frame to gray `gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)`
* then we use cv2.imshow() to show the frame `cv2.imshow('frame',gray)`
* we add escape logic
```
if cv2.waitKey(1) & 0xFF == ord('q'):
	break;
```
* we then have to stop capturing `cap.release()`
* and then destroy the window `cv2.destroyAllWindows()`
* the whole code looks like
```
import cv2

cap = cv2.VideoCapture(1)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',gray)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```
* we run it and it works!!!!!!!!!!
* we will play around a bit... processing the frame
* we want to be able to save the stream to a file
* we need a writer object `writer = cv2.VideoWriter('./myVideo1.mp4',cv2.VideoWriter_fourcc(*'XVID'),30,(width,height))` we use the cv2.VideoWriter() method to save to a file. it takes 4 arguments
	* the filepath of the file to write
	* the 4-BYTe code specing the codec to be used (different per operating system)
	* the fps to use (frames per second) . we can see our cameras FPS with cv2.CAP_PROP_FRAME_COUNT
	* the size of the frame(wifdth,height)
* in the while loop after reading we write the frame to the file `writer.write(frame)`
* after exiting we release the writer `writer.release()`

### Lecture 37 - Using Video Files

* in the previous lecture w esaw hpw tp stream and use the video captured by a camera.
* we will now see how to use existing video files
* we work on anotebook single cell
* we import  cv2
* we will read mp4 files from disk. its the same as capturing from camera. we just instead of index provide filepath `cap = cv2.VideoCapture('../DATA/hand_move.mp4')`
* if we insert wrong filename opencv does not exit or codec is  not supported. it just streams nothing. we put a helpful check
```
if cap.isOpened() == False:
    print('ERROR FILE NOT FOUND OR WRONG CODEC USED')
```
* our while loop is based on cap.isOpened()
* we read a frame. if we have red sthing the we show it and listen  for exit key
* if we get no frame we braeak th e while loop
* we cleanup capture and window
```
cap = cv2.VideoCapture('../DATA/hand_move.mp4')

if cap.isOpened() == False:
    print('ERROR FILE NOT FOUND OR WRONG CODEC USED')
    
while cap.isOpened():
    
    ret, frame = cap.read()
    
    if ret == True:
        
        cv2.imshow('frame',frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
```
* video plays very fast. openCV is not built for presenting videos but processing them so its fast
* to present the video at human normal speed we import time `import time`
* we add a sleep time equal to the frame rate : for 20fps => 50ms in the while loop `time.sleep(1/20)`

### Lecture 38 - Drawing on Live Camera

* drawing onm video is similar with drawing on image  (frame==image)
* we import cv2
* we start capture from camera `cap = cv2.VideoCapture(0)`
* we get width and height of caputer frame
```
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
```
* we ll draw a rectangle on the stream
* we first get the 1/2 of frame dimensions `x = width // 2` // gets the integer part of the division
* we set the width of the rect `w = width // 4`
* we draw the rect on frame `cv2.rectangle(frame,(x,y),(x+w,y+h),color=(0,0,255),thickness=4)`
* we show the frame `cv2.imshow('frame',frame)`
* we add escape logic and cleanup
* to interactively draw on the video we use the capture-show frame boilerplate.
* we add a callback to modify the global vars
```
def draw_rect(event,x,y,flags,paran):
    global pt1,pt2,topLeft_clicked,botRight_clicked
    
    if event == cv2.EVENT_LBUTTONDOWN:
        
        # RESET THE RECTANGLE (IT CHECKS IF THE RECT THERE)
        if topLeft_clicked == True and botRight_clicked == True:
            pt1=(0,0)
            pt2=(0,0)
            topLeft_clicked = False
            botRight_clicked = False
            
        if topLeft_clicked == False:
            pt1 = (x,y)
            topLeft_clicked = True
        
        elif botRight_clicked == False:
            pt2 = (x,y)
            botRight_clicked = True
```
* we set some global vars
```
pt1=(0,0)
pt2=(0,0)
topLeft_clicked = False
botRight_clicked = False
```
* we do the connection to the callback
```
cv2.namedWindow('Test')
cv2.setMouseCallback('Test',draw_rect)
```
* we draw the rectangle in the while loop using the global values
```
    if topLeft_clicked:
        cv2.circle(frame,center=pt1,radius=2,color=(0,0,255),thickness=-1)
    if topLeft_clicked and botRight_clicked:
        cv2.rectangle(frame,pt1,pt2,(0,0,255),3)
```

### Lecture 39 - Video Basics Assessment

* 

## Section 6 - Object Detection with OpenCV and Python

### Lecture 41 - Introduction to Object Detection

* The Section Goals
	* Understand a variety of object detection methods
	* we ll build up on more complex methods as we go along
* Template Matching
	* simply looking for an exact ccopy of an image on another image
* Corner Detection (General Detection)
	* looking for corners in images
* Edge Detection (General Detection)
	* expanding to find general edges of objects
* Grid Detection (General Detection)
	* combining  both concepts to find grids in images (useful for applications)
* Contour Detection
	* Allows us to detect foreground vs Background images
	* Also allows for detection of external vs internal contours (e.g grabbing the eyes and smile from a cartoons smile face)
* Feature Matching
	* more advanced methods of detecting matching objects in another image, even if the target image is not shown exactly the same in the image we are searching
* Watershed Algorithm
	* Advanced algorithm that allows us to segment images into foreground and background
	* also allows us to manually set seeds to choose segments of an image
* Facial and Eye Detection
	* We will use Haar Cascades to detect faces in images
	* Note that this is not yet facial recognition that requires deep learning which we will learn in future section
* Project Assessment
	* A computer vision app that can blur licence plates automatically

### Lecture 42 - [Template Matching](https://en.wikipedia.org/wiki/Template_matching)

* Template matching is the simplest form of object detection
* it simply scans a larger image for a provided template by sliding the template target image accross the larger image
* we are talking for an almost exact match
* the main option that canbe adjusted is the comparison method used as the target template is slid across the larger image
* the methods are some sort of correlation based metric
* cv2 offers various methods (TM_SQDFF sqaure difference) (TF_SQDIFF_NORMED normalized square difference) etc
* we start a notebook with usual imports
* we imread teh full image we will search in (/sammy.jpg) and color correct it
* we imread a subset of the full iamge 'sammy_face.jpg' and color correct it
* the subset is a crop of the full image
* template matching is part pointless wecause we know beforehand
* we will use the eval function. eval like sum is a builtin python method. it evaluates a string for a function call
```
sum([1,2,3])
>> 6
mystring = 'sum'
myfunc = eval(mystring)
myfunc([1,2,3])
>> 6
```
* for ease of evaluation we put all TM avaialble methods of cv in an array and loop over it `methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR','cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']`
* we loop over the array `for m in methods:`
* first thing we make a copy of the full image `full_copy = full.copy()`
* we make a method to use using eval `method = eval(m)`
* we do the actual template matching using cv2.matchTemplate() `es = cv2.matchTemplate(full_copy,face,method)` passing full image, the tempalte and the method
* result is a heatmap (we see it if we plt.imshow(res)). it gives higher values on where it thinks it found the best match. the max value is the best fit (correlation)
* we will use the min and max val of the heatmap and their locations to draw a rect around the match. we use cv2.minMaxLoc(). `min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(res)`
* as SQDIFF works the opposite minval == best corr
```
    if method in [cv2.TM_SQDIFF,cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
```
* we get bottom right of rect to draw the match area from template shape
```
    height, width,channels = face.shape
    bottom_right = (top_left[0]+width,top_left[1]+height)
```
* we draw the rect `cv2.rectangle(full_copy,top_left,bottom_right,(255,0,0),10)`
* we plot and show the image
```
    plt.subplot(121)
    plt.imshow(res)
    plt.title('HEATMAP OF TEMPALTE MATCHING')
    plt.subplot(122)
    plt.imshow(full_copy)
    plt.title('DETECTION OF TEMPLATE')
    plt.suptitle(m)
    
    plt.show()
```
* 'plt.show()' helpt the iteration so we dont overwrite images
* we see the results. only TM_CCORR performs badly

### Lecture 43 - Corner Detection - Part One - [Harris Corner Detection](https://en.wikipedia.org/wiki/Harris_Corner_Detector)

* when thinking about corner detection in computer vision, we should define what is a corner
* a corner is a point whose local neighborhood stands in two dominant and different edge detections. it can be interpreted as the junction of two edges, where an edge is a sudden change in image brightness
* we will look at 2 of the most popular algorithms for corner detection.
	* Harris Corner Detection
	* Shi-Thomas Corner Detection
* Harris Corner Detection
	* 1988 by Chris hharris and Mike Stevens
	* the basic intuition is that corners can be detected by looking for significant change in all directions
	* shifting a window in any direction on a corner region will result in a large change in appearance
	* doing the same on aflat region will have no effect at all
	* doing shifting on an edge wonth have major change if we shift along the direction of the edge
	* In a nutshell Harris Corner Detection math says: if we scan the image with a window (like we did with kernels) and we notice an area where there is major change no matter in which direction we scan, we expect a corner to be there. the window does shifting
* Shi-Thomasi Corner Detection
	* 1994 by J.Shi and C.Tomasi in the papaer Good Features to Track
	* It made a small mod to the Harris Corner Detection that geve better results
	* the mod is a change to the scoring function selection criteria that Harris uses for corner detection: Harris uses R=λ1λ2-κ(λ1+λ2) Shi-Tomasi uses  R=min(λ1,λ2)
* We ll explore how to use both with the OpenCV lib.
* we do normal imports in notebook
* we imread a chessboard image 'flat_chessboard.png' and color correct it.
* image is a perfect grid with clear corners and clear edges
* we cnv it to grayscale
* we also read a real chess image 'real_chessboard.jpg' we expect the algo to find corners related to pieces as well. we color correct it and turn it to grayscale
* we apply harris corner detection to the flat nd real chess image
* first we convert the grayscale image (0-255) to float vals (0. to 1.) we do it with plain casting `gray = np.float32(gray_flat_chess)`
* we then apply harris cd `dst = cv2.cornerHarris(src=gray,blockSize=2,ksize=3,k=0.04)` passing in:
	* src image
	* blocksize of the window
	* ksize of the sobel operator used for edge detection
	* k param (harris detector free param) (typically 0.04)
* we then dilate results for ploting it `dst = cv2.dilate(dst,None)`
* there is a threshold for optimal value that varies upon the image. we choose it to be 0.01*max() value of resutl and use it to turn to red the pixels that are over the threshold in terms of corner detection rerult. we apply it to the original image with numpy array indexing `flat_chess[dst> 0.01*dst.max()] = [255,0,0]`
* we plot the image. detection is perfect
* outer edges are not detected. it is seen as flat space.
* we will apply haris to the grayscale version of the real chess
```
gray = np.float32(gray_real_chess)
dst = cv2.cornerHarris(src=gray,blockSize=2,ksize=3,k=0.04)
real_chess[dst> 0.01*dst.max()] = [255,0,0] #RGB
plt.imshow(real_chess)
```
* it detec alot of corners of pieces as well

### Lecture 44 - Corner Detection - Part Two - [Shi-Tomasi Detection](http://www.ai.mit.edu/courses/6.891/handouts/shi94good.pdf)

* we will use the same images for testing (real,flat + gray versions)
* we use 'cv2.goodFeaturesToTrack' method with params:
	* src image
	* max corners we want returnd (0 to return all)
	* a quality level param (minimum eigen val)
	* minimu distance 
* we will draw little circles in positions he thinks he found the corners. it does not store the points of corners like the harris . we need to flatten out the arrya and draw circles on it
* corners are float so we turn them to int `corners = np.int0(corners)`
* the we iterate through corners getinng coord from flatened rray and drawing circle
```
for i in corners:
    x,y = i.ravel()
    cv2.circle(flat_chess,(x,y),3,(255,0,0),-1)
```
* we then plot the drawun image
* we do the same for real_chess trying to detect 100 corners
```
corners = cv2.goodFeaturesToTrack(gray_real_chess,100,0.01,10)
corners = np.int0(corners)
for i in corners:
    x,y = i.ravel()
    cv2.circle(real_chess,(x,y),3,(255,0,0),-1)
plt.imshow(real_chess)
```
* we see better results than harris

### Lecture 45 - [Edge Detection](https://en.wikipedia.org/wiki/Edge_detection)

* In this lecture we will learn how to use the [Canny Edge Detector](https://en.wikipedia.org/wiki/Canny_edge_detector) one of themost popular edge detection algorithms
* It was developed in 1986 by John Canny and is a multi-stage algorithm
* Canny Edge Detection Pipeline:
	* Apply Gaussian filter to smooth the image in order to remove the noise
	* Find the intensity gradients of the image
	* Apply non-maximum suppression to get rid of spurious response to edge detection
	* apply double threshold to determine potential edges
	* track edge by hysteresis: Finalize the detection of edges by suppressing all the other edges that are weak and not connected to strong edges
* For high res images where we only want general edges, it is usually good idea to apply a custom blur before applying canny algorithm
* Canny algorithm requires the user to decide on low and high threshold values
* In our notebook we provide an equation for picking a good starting point for threshold vals, but often we will need to adjust to our particular image
* we add the normal imports
* we will work on sammy_face.jpg. we dong care about color correction
* we will apply the canny edge detector straight through (no blurring) `edges = cv2.Canny(image=img,threshold1=127,threshold2=127)` we set low and hight threshold to half. we plot and see there is a lot of noise in the result
* we can solve it with:
	* blurring the image beforehand
	* play with threshold
* we play with threshold and get some good results with `edges = cv2.Canny(image=img,threshold1=220,threshold2=240)`
* we will use a formula that helps select good thresholds
* we calculat ethe median pixel val `med_val = np.median(img)` its 64
* we select the thresholds based on
```
# LOWER THRESHOLD TO EITHER 0 OR 70% OF THE MEDIAN VAL, WHICHEVER IS GREATER
lower = int(max(0,0.7*med_val))
# UPPER THRESHOLD TO EITHER 1300% OF THE MEDIAN VAL oR 255, WHICHEVER IS SMALLER
upper = int(min(255,1.3*med_val))
```
* we apply the thresholds `edges = cv2.Canny(image=img,threshold1=lower,threshold2=upper)` results are actually worse
* we blur `blurred_img = cv2.blur(img,ksize=(5,5))` and apply result is considerably better
* to improve more we increase kernel size

### Lecture 46 - Grid Detection

*  often cameras can create a distortion in an image, such as radial distortion and tangential distortion
* a food way to account for these distortions when performing operation like object tracking is to have a recognizable pattern attached to the object being tracked
* grid patterns are often used to calibrate cameras and track motion (eg attach/draw a cube on a grid that will move as grid moves)
* openCV has built in methods for tracking grids and chessboard like patterns
* we do normal imports
* we read the 'flat_chessboard.png' image
* for grid detection to work the grid has to have a chessboard like appearance. then we have to place it on the samera we want to calibrate
* to find the chessboard corners we use cv2.findChessboardCorners `found,corners = cv2.findChessboardCorners(flat_chess,(7,7))` we pass in the src image and a tuple with the number of grid edges the grid area as in each direction. it returns a tuple with found (a bool if it found the pattern) and the position of the grid edges
* corners is a list of coordinates 
* we use them with another build in method 'cv2.drawChessboardCorners' `cv2.drawChessboardCorners(flat_chess,(7,7),corners,found)` which draws on the image we pass the corners found
* another grid like pattern is circle based grids (dot grids)
* we read in 'dot_grid.png' with perfectly clean circles
* we will use equivalent cv2 methods like chessboard but for circlegrid (same concept same params) `found,corners = cv2.findCirclesGrid(dots,(10,10), cv2.CALIB_CB_SYMMETRIC_GRID)` we use alsoa a grid param
* corners are in a same format. we use the drawchessboardcorners passing the corners `cv2.drawChessboardCorners(dots,(10,10),corners,found)`
* grid detection is used for camera calibration

### Lecture 47 - Contour Detection

* Contours are defined as simply a curve joining all the continuous points (along the boundary), having same color or intensity
* Contours are a useful tool for shape analysis and object detection and recognition
* OpenCV has a built-in Contour finder finder function that can also help us differentiate between internal and external contours
* we do the normal imports
* we read in (in grayscale) an image 'internal_external.png' with simple contours (internal and external)
* to extract the contours we use cv2.findContours `image, contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)` with has as arguments
	* the src image,
	* the type of detection (internal,external contours, both)
	* the algorithm to be used
* the function returns the image. a list (conttours) and a 3d array with hierarchies (onyly if we use both internal and external.
* the contrours are 22 whic matches what we expect forthis simple image
* to show the contours we initialize a black image as zeros `external_contours = np.zeros(image.shape)`
* then we iterate in the list of contours and:
* if the 3rd element of the hierarchy array for the indexed contour is -1 it means it is an external so we draw it using the drawContours method that takes the black image (where to draw on) the contours object, the index the colour it will use and the thicknes (-1  for fill)
```
for i in range(len(contours)):
    
    # EXTERNAL CONTOUR
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(external_contours,contours,i,255,-1)
```
* internal contour edges touch the foreground to get them and draw them we use the same loop but the condition is the hierarchy to be != -1 (non external)

### Lecture 48 - Feature Matching - Part One

* This is the halfway of the course.
* So far we ve learned a lot of technical syntax, but havent really neen able to apply it to more complex computer vision applications
* This is the point where we begin to do useful computer vision apps. we will use all our technical knowledge and python sysntax skills with OpenCV to create programs that are directly applicable to realistic situations
* We will begin with Feature Matching
* We ve already seen template matching to findobjects (tempalte images) within a larger iamge. it required an exact copy of the image.
* usually this is not useful in real world situations as we ll have an indicative image of what we are looking for, not an exact copy.
* what we do in such situations is feature matching
* Feature matching extracts defining key feats from an input image (using ideas from corner,edge, and contour detection)
* then using a distance calculation finds all the matches in a secondary image
* this means we are no longer required to have an exact copy of the target image in the secondary image
* We will use 3 methods:
	* Brute-Force matching with ORB Descriptors
	* Brute-Force Matching with SIFT Descriptors and Ratio Test
	* FLANN based Matcher
* we will test with a generic cereal box image and see if we can find its matching box in the cereal isle
* we do normal imports and use our display helper function
* we imread a cereal image 'reeses_puffs.png' as grayscale. this is the query image
* the target image is 'many_cereals.jpg' we imread it as a grayscale. in target image the query image exists but not as exact copy
* First we apply Brute Force Detection with ORB descriptors
* we first create the detector `orb = cv2.ORB_create()`
we then apply the detector on both images (target and query) to extract keyponts and descriptors (Nonoe stands if we want to use a mask)
```
kp1, des1 = orb.detectAndCompute(reeses,None)
kp2, des2 = orb.detectAndCompute(cereals,None)
```
* the we apply brute force matcher using default params `bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)`
* we use the bruteforce result to get the matches between descriptors from 2 images `matches = bf.match(des1,des2)`
* we sort the matches based on their distance attribute (represents the match) `matches = sorted(matches, key=lambda x:x.distance)`
* we use a convenience method to draw matches for showing  using images, keypoints and matches array. `reeses_matches = cv2.drawMatches(reeses,kp1, cereals,kp2, matches[:25], None,flags=2)`
* we then display the image. we see that we have no successful match

### Lecture 49 - Feature Matching - Part Two

* we will now use SIFT (scale invariant feature transform) descriptors fro bruteforce feat detection
* it performs better in cases where query image is scaled in the target image
* we start with creating a sift object `sift = cv2.xfeatures2d.SIFT_create()`
* in the same way as before we extract keypoints and descriptors from both query and target image.
```
kp1, des1 = sift.detectAndCompute(reeses,None)
kp2, des2 = sift.detectAndCompute(cereals,None)
```
* we have calculated the descriptors we will compare them using brute force `bf = cv2.BFMatcher()`
* we will calc the matches fromthe bf object in a different manner `matches = bf.knnMatch` what this does is takes 2 sets of descriptors and a value k (number of best matches  it will find per descriptor of the query set)
* descriptors are coordinates of where feats were found
* as i set 2 as k the matches obj is an arraw of 2 matchobjects per descriptor. first match is better than the second
* we will now apply a ratio test. our intution is that if the distance of match1 is close to the distance of match2 the this descriptor's  feat is probably a good match between query set and target set
```
good = []
for match1,match2 in matches:
    # IF MATCH 1 DISTANCE IS <75% OF MATCH2 DISTANCE
    # THEN DESCRIPTOR WAS A GOOD MATCH, KEEP IT
    if match1.distance < 0.75*match2.distance:
        good.append([match1])
```
* this filtering does pretty good job (keeps ~5%)
* we draw the matches using conv method `sift_matches = cv2.drawMatchesKnn(reeses,kp1,cereals,kp2,good,None,flags=2)`
* we display. resutls are actually very good
* we ll work with FlANN (Fast Library for Aproximate Nearest Neighbours) based matcher. its much faster than Bruteforce but it finds general good matches
* we can play with FLANN params to imporve resutls but it slows down the algo
* we start like before creating a sift object `sift = cv2.xfeatures2d.SIFT_create()` and getting keypoints and descriptors from images
* we set Flann params (to defaults)
```
FLANN_NDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_NDEX_KDTREE,trees=5)
search_params = dict(checks=50)
```
we compare descriptors with flann `flann = cv2.FlannBasedMatcher(index_params,search_params)`
* we grab the k nearest neigbours matches with `matches = flann.knnMatch(des1,des2,k=2)`
* we do a ratio test like before
```
good = []
# LESS DISTANCE == BETTER MATCH
for match1,match2 in matches:
    # IF MATCH 1 DISTANCE IS <75% OF MATCH2 DISTANCE
    # THEN DESCRIPTOR WAS A GOOD MATCH, KEEP IT
    if match1.distance < 0.75*match2.distance:
        good.append([match1])

```
* we use draw helper to draw them `sift_matches = cv2.drawMatchesKnn(reeses,kp1,cereals, kp2,good, None, flags=2)`
* we display and see the result. we get good results with increase in speed
* if we use flags=0 we see also the dots of the matches (potential feats to match on).
* to play with coloring in presentation to make understanding better we will mask the matches.
* after geting them and before ratio test we do `matchesMask = [[0,0] for i in range(len(matches))]` so we get array of nested size 2 zero arrays equal in num to matches
* we will turn them on (0to1) if we have a good match (during ratio test)
```
# LESS DISTANCE == BETTER MATCH
for i,(match1,match2) in enumerate(matches):
    # IF MATCH 1 DISTANCE IS <75% OF MATCH2 DISTANCE
    # THEN DESCRIPTOR WAS A GOOD MATCH, KEEP IT
    if match1.distance < 0.75*match2.distance:
        matchesMask[i]=[1,0]
```
* we no longer need to gopy in good. we use indexing in mask
* we create a dray params dict obj `draw_params = dict(matchColor=(0,255,0),singlePointColor=(255,0,0),matchesMask=matchesMask,flags=0)`
* our draw method becomes `flann_matches = cv2.drawMatchesKnn(reeses,kp1,cereals,kp2,matches,None,**draw_params)`

### Lecture 50 -[ Watershed Algorithm](https://en.wikipedia.org/wiki/Watershed_(image_processing)) - Part One

* In geograpshy, a watershed is a land area that channels rainfall and snowmelt to creeks,streams and rivers and eventually to outflow points such as reservoirs, bays, and the ocean
* These watersheds can then be segmented as topolographical maps with boundaries (topographical lines of altitude in maps)
* Metaphorically the watershed algorithm transformation treat the image it operates upon like a topographic map, with the brightness of each point representiong its height, and finds the lines that run along the tops of ridges (like water aggreages in geography, brightness aggregates in images)
* Any grayscale image can be viewed as a topographic surface where high intensity denotes peaks and hills while low intensity denotes valleys
* The algorithm can then fill every isolated valleys (local minima) with different colored water (labels)
* as "water" (inensity) rises, depnding on the peaks (gradients) nearby, "water" from different valleys (different segments of the image) with different colors could start to merge
* To avoid this erging, the algorithm creates barriers (segment edge boundaries) in locations where "water" merges
* this algorithm is especially useful for segmenting images into background and foreground in situations  that are difficult for other algorithms
* a common example is the use of coins next to ech other on a table. for most CV algos when they see the image the coinsed all cois as a large blob
* it may be unclear to the algo if it should be treated as one large object or many small objects.
* watershed algo can be very effective for these sort of problems
* later on we will also learn how to provide our own custom 'seeds' that allow  us to manually start where the valleys of the watersheds go
* we ll begin exploring the syntax of te watershed algorithm with OpenCV and then expand this idea to set our own seeds
* We start our notebook with the normal imports and the helper method
* we imread 'pennies.jpg' a high res image of 6 coins attached to each other
* our goal is to be able to produce 7 segments in the image (6 for coins and 1 for background)
* we will test algos we know sofar to show the waekness of the algos in distinguising the coins
* We first apply median blur to get rid of feats we dont need `sep_blur = cv2.medianBlur(sep_coins,25)`
* we will turn it to grayscale `gray_sep_coins = cv2.cvtColor(sep_blur,cv2.COLOR_BGR2GRAY)`
* We will apply binary threshold `ret,sep_coins_thresh = cv2.threshold(gray_sep_coins,160,255,cv2.THRESH_BINARY_INV)` wee that no matter how we play the coins are allways attached in one shape (we could erode)
* We will find the contours `image,contours,hierarchy = cv2.findContours(sep_coins_thresh.copy(), cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)`
* we draw the external contours
```
for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(sep_coins,contours,i,(255,0,0),10)
```
* contour is one giant contour
* we need a more advanced method

### Lecture 51 - Watershed Algorithm - Part Two

* we read the same image
* we apply median blur (huge kernel for huge image) `img = cv2.medianBlur(img,35)`
* we turn to grayscale `gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)`
* we try thresholding `ret, thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)` we see that despite the high blur we still pick up features in binary thresh
* We will apply Otsu's method for thresholding with beteer results which is a very good match to the watershed algorithm. we apply again thersholiding adding OTSU (otsu works with full range between low and up treshold) `ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)`
* our circles are still connected
* we do noise removal (no effect in such a simple image. makes sense for real complex images) using the morphological operators `opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=2)` for this image has no effect
* we still face the problem. we have one blob as object
* we get the sure background area by dilating onthe opened image `sure_bg = cv2.dilate(opening,kernel,iterations=3)`
* what we need to do for the watershed is to set seeds that we are sure that are in the foreground (6 seeds 1 per coin)
* how we grab things we are sure are in the foreground vs things in the background? we use [distance transform](https://en.wikipedia.org/wiki/Distance_transform). what this dows is as we move away from boundaries with background (0) pixels get higher values (become brighter). we can apply this to our thresholded image and expect the coins center to be the brightest points. then we can rethreshold and get the 6 seed points `dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)`
* we reapply thresholding to get surely foreground points `ret,sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)` 0.7*dist_transform.max() is a typical value used. we get 6 points we are sure to be in foreground. we will use them as seeds
* the region outside the dots is the unknown region. we need the watershed algo to figure out what it is
* we get the unknown region by subtracting sure foreround from sure background region. that where we need to use watershed algo
```
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
```
* we need to add label markers to the 6 points and let them be the seeds of watershed algo
* we get the markers from foreground `ret, markers = cv2.connectedComponents(sure_fg)` and add 1 to separate rom background. markers are the same points but with different color
* we explicitly set the unknown area to 0 `markers[unknown==255] = 0` thats why we added 1 before. to clearly  separate it from unknown area
* we are now ready to fill/flood the unknwon area with watershed algrithm.. `markers = cv2.watershed(img,markers)` we have clear separation.
* we cna now confidently get the contours
```
image,contours,hierarchy = cv2.findContours(sep_coins_thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)`
for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(sep_coins,contours,i,(255,0,0),10)
```

* we did the work manually. in next lecture we will do it autom.

### Lecture 52 - Custom Seeds with Watershed Algorithm

* we want to be able to click on the image setting the seeds manually. and let the algo run all the steps to do the segmentation automatically
* in a new notebook we do the normal imports
* we read in an image 'road_image.jpg' and do a copy out of it
* we wont do color correction as we will use opencv to view and interact with the image
* we create an empty space to draw on the results of the algorithm using the shape of the road. one for the markers for watershed `marker_image = np.zeros(road.shape[:2],dtype=np.int32)` and one to draw the segments `segments = np.zeros(road.shape,dtype=np.uint8)`
* we then have to choose how to create the colors for the markers. we will use colormaps
* matlplotlib colormaps have qualitatibe colormaps that are indexable
```
from matplotlib import cm
cm.tab10(0)
```
* what we get is a color in tuple form with rgb vals in float format + alpha param
* to use the colors we cast them to tuple `tuple(np.array(cm.tab10(0)[:3])*255)` color is in OpenCV BGR format
* we make it a func	
```
def create_rgb(i):
    tuple(np.array(cm.tab10(i)[:3])*255)
```
* and use it to create 10 distinct colors for markers
```
colors = []
for i in range(10):
    colors.append(create_rgb(i))
```
* we start implementing our application
* we define the globals
```
current_marker = 1 # color choice
marks_updated = False # markers updated by watershed algorithm
```
* we write our callback function. it mod the global marks_updated. it listens to event LBUTTONDOWN.when this happens it draws a circle ont he road_copy the user sees and draws a marker on the marker_image to be fed to the algorithm
```
def mouse_callback(event,x,y,flags,param):
    global marks_updated
    if event == cv2.EVENT_LBUTTONDOWN:
        # MARKERS PASSED TO THE WATERSHED ALGO
        cv2.circle(marker_image,(x,y),10,(current_marker),-1)
        # USER SEES ON THE ROAD IMAGE
        cv2.circle(road_copy,(x,y),10,colors[current_marker],-1)
        marks_updated = True
```
* we add window and callback bind for Riad Image window 
```
cv2.namedWindow('Road Image')
cv2.setMouseCallback('Road Image', mouse_callback)
```
* we start the while loop
	* we show 2 windows: one for placing the markins on road image and one for segments
	* we add escape logic
	* we add reset logic (reseting markers sergments matrices and current marker selection)
	* we add logic to change marker group with digits
	* when user clicks (marker update)
		* we make a copy of marker image
		* we run watershed on original image based on marker image copy (current selection)
		* we rest segments
		* we redrow them  based ont the markers_mage copy status (watershed output)
```
while True:
    
    cv2.imshow('Watershed Segments',segments)
    cv2.imshow('Road Image',road_copy)
    
    # CLOSE ALL WINDOWS
    k = cv2.waitKey(1)
    
    if k == 27:
        break
    
    # CLEARING ALL COLORS IF USER PRESSES C KEY
    elif k == ord('c'):
        road_copy = road.copy()
        marker_image = np.zeros(road.shape[:2],dtype=np.int32)
        segments = np.zeros(road.shape,dtype=np.uint8)
    
    # UPDATE COLOR CHOICE
    elif k > 0 and chr(k).isdigit():
        current_marker = int(chr(k))
        
    # UPDATE THE MARKINGS
    if marks_updated:
        
        marker_image_copy = marker_image.copy()
        cv2.watershed(road,marker_image_copy)
        
        segments = np.zeros(road.shape,dtype=np.uint8)
        
        for color_ind in range(n_markers):
            # COLORING THE SEGMENTS, NUMPY CALL
            segments[marker_image_copy==(color_ind)] = colors[color_ind]
    
cv2.destroyAllWindows()
```

### Lecture 53 - Introduction to [Face Detection](https://en.wikipedia.org/wiki/Haar-like_feature)

* In this lecture we will explore face detection using Haar Cascades, which is key component of the Viola-Jones object detection framework
* Keep in mind we are talking about face detection NOT face recognition
* we will be able to very quickly detect if a face is in an image and locate it
* however we wont know who's face it belongs to.
* we would need a  really large dataset and deep learning for facial recognition
* In 2001 Paul Viola and Michael Jones published their method of face detection based oin the simple concept of a few key features
* They also came up with the idea of precomputing an integral image to save time on calculations
* Lets understand the main feature types Viola and jones Proposed
* main feature types are:
	* edge features (horizontal ore vertical) e.g [[0,0,0],[1,1,1]]
	* line features (horizontal or vertical) e.g [[1,1,1],[0,0,0],[1,1,1]]
	* four rectangle features e.g [[0,1],[1,0]]
* each feature is a single value obtained by subtracting sum of pixels under white rectangle from sum of pixels under black rectangle
* realistically, our images won't be perfect edges or lines
* these feats are calculated by subtracting the mean of the dark region from the mean of the light region
* a perfect line (0,1) whoud result to an 1. the closer our result (delta) is to 1 the better the feature
* we then set a threshold above which we consider to have a feature
* calculating these sums for the whole image can be computational expensive
* the Viola-Jones algorithm solves it by using the precalculated integral image
* this results in an O(1) running time of the algorithm
* an integral image is known as a summed area table which comes fromt the original image by summing the pixel values in the area defined as a rectanngle with topleft =(0,0) and pt2=(x,y) the position of the pixel (bottom right)
* this allows to calculate very fast the mean and the delta
* this algorithm also saves time by going through a cascade of classifiers
* this means we will treat the image to a series (a cascade) of classifiers based on the simple feats we saw earlier
* once an image fails a clasifier we can stp attempting to detect a face
* a common misconception behind face detection with this algo is that the algorithm slowly scans the entire image looking for a face
* this would be very inefficient. instead we pass a cascade of classifiers
* first we need a front face image of a persons face
* then we turn it to grayscale
* then we will begin to search for the Haar Cascade features
* one of the very first features searched for is an edge feature indicating eyes and cheeks
* if it passes we go to next feature such as the bridge line of the nose
* if it passes we continue with other features (eyebrows edges, mouth line etc)
* untill the lagorithm decides it has detected a face based on the features
* theoretically this approach can be used for a variety of objects or detections (like pretrained eye detector)
* the downside of this algorithm is that very large datasets are needed to create our own feature sets
* luckily many pre-trained sets of features exist
* OpenCV comes wwith pre-trained xml files of various Haar Cascades
* Later on in the deep learning section of the course we will see how to create our own classification algorithm for any distinct group of images (e.g cats vs dogs)
* we have placed pre-trained .xml files int eh DATA folder
* we will also be using a pre-trained file for our upcomming project assessment
* first we ll expore how to use facial detection with OpenCV

### Lecture 54 - Face Detection with OpenCV

* we do the normal imports
* we will use two portrait images. one is professionally edited with gradients causing issues down the line
* also will use a group photo in grayscale
* we imread the images
* we need to create a classifier and pass in teh XML classifier `face_cascade = cv2.CascadeClassifier('../DATA/haarcascades/haarcascade_frontalface_default.xml')`
* we will functionalize the way cascades work 
```
def detect_face(img):
    face_img = img.copy()
    face_rects = face_cascade.detectMultiScale(face_image)
    for (x,y,w,h) in face_rects:
        cv2.rectangle(face_img,(x,y),(x+w,y+h),(255,255,255),10)
    return face_img
```
* in this func we make a copy of image and use detectMultiScale on the cascade object passing in the image
* what it returns is an array of rectangles (topleft position   width and height)
* we iterate n teh array drawing the rectangles on teh image and return it
* we test it on our test images . it works but for the multifface image it throws false positives
* we will adjust some params to imrpove performance (scale factor and minimum neighbors) `face_rects = face_cascade.detectMultiScale(face_img,scaleFactor=1.2,minNeighbors=5)`
* we test and we have a false negative of a face not lookng in the camera
* we will look for eyes using an Eye_cascade file. in nadia it works but not in denis
* we will do it with video capture

### Lecture 55 - Detection Assessment

* 

## Section 7 - Object Tracking

### Lecture 57 - Introduction to Object Tracking

* Object Tracking Section Goals
	* Lear basic object tracking techniques: Optical Flow, MeanShift and CamShift
	* Understand more advanced tracking: Review Built-in Tracking APIs

### Lecture 58 - [Optical Flow](https://en.wikipedia.org/wiki/Optical_flow)

* Optical Flow is the pattern of apparent motion of image objects between two consecutive frames caused by the movements of the object or camera
* Optical flow analysis gas a few assumptions
	* the pixel intensities of an object do not change between consecutive frames
	* neighbouring pixels have similar motion
* The optical methods in OpenCV will first take in a given set of points and a frame
* Then it will attempt to find those points in the next frame
* It is up to the user to supply the points to track
* we consider a five frame clip of a ball moving up and towards the right
* given the clip we cannot determine if the ball is moving or if the camera moved down and to the left
* using OpenCV we pass in the previous frame, previous points and the current frame to the Lucas-Kanade function
* the function attempts to locate the (tracked) points in teh current frame
* The Lucas-Kanade computes optical-flow for a sparse feature set (meaning only the points it was told to track)
* But what if we want to track all teh points in teh video
* In that case we can use Gunner Farnerback's algorithm (also built in to OpenCV) to calculate dense optical flow
* This dense optical flow will calculate flow for all points in an image
* It will coler them black if no flow (no movement) is detected

### Lecture 59 - Optical Flow Coding with OpenCV - Part One

* We start with Lucas-kanade method for sparce flow
* we will use simple SHi-Tomashi corner detection to find point to track
	* we set the params as dict `corner_track_params = dict(maxCorners= 10, qualityLevel=0.3, minDistance=7, blockSize=7)`
	* we will find 10 corners in first frame and track them
* we se LK params as dict passing some default vals: 
	* winSize = window size (smaller more sensitive to noise and might lose larger motions, larger window might miss small motions), 
	* maxLevel = (LK uses image pyramide for image proc) is the levels of pyramid used. what we gain is we can track motions at various resolutions of the image
	* criteria = 2 criteria (maximum num of iterations, epsilon or accuracy) we should adjust them depending on the video
* we grab a frame from camera to find the poits to track and turn it to grayscale
```
cap = cv2.VideoCapture(1)
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame,cv2.COLOR_BGR2GRAY)
```
* we get the points to track `prevPoints = cv2.goodFeaturesToTrack(prev_gray,mask=None,**corner_track_params)
`
* we create a mask to draw the points and create lines on the video (initialize with 0) `mask = np.zeros_like(prev_frame)`
* we dd the while loop where
	* we capture a frame and turn it to grayscale
	* we calcualte the optical flow with LK `nextPts, status, err =  cv2.calcOpticalFlowPyrLK(prev_gray,frame_gray,prevPoints,None,**lk_params)` we pass inL
		* prev frame
		* curr frame
		* prevPoints
		* None nextPoints (we will get them as return vals)
		* the config params
	* we will use the returned status array which outputs a status vector where each element of the vector is set to 1 if the flow for teh correspondent feature has been found
	* we use it to get the good new points `good_new = nextPts[status==1]`
	* also to get the good prev poitns (for drawing the line) `good_prev = prevPoints[status==1]`
	* we zip both iterating through to draw the lines and set the dots of point son frame
```
    for i, (new,prev) in enumerate(zip(good_new,good_prev)):
        x_new, y_new = new.ravel()
        x_prev, y_prev = prev.ravel()
        
        mask = cv2.line(mask,(x_new,y_new),(x_prev,y_prev),(0,255,0),3)
        
        frame = cv2.circle(frame,(x_new,y_new),8,(0,0,255),-1)
```
* we mask the frame and show it 
```
    img = cv2.add(frame,mask)
    cv2.imshow('tracking',img)
```
* we set durr frame as prev and good new points as prev for next iteration (frame)
* we need to reshape the good_new points so its accepted by the LK
* we release frame and destroy window

### Lecture 60 - Optical Flow Coding with OpenCV - Part Two

* We will take the entire image to detect points
* to see all the params we can see the course notebook
* we do usual imports
* we start capture object
* we read a frame a frame (initial frame) and turn it to grayscale
* we setup an HSV based mask `hsv_mask = np.zeros_like(frame1)`
* we set saturation to max `hsv_mask[:,:,1] = 255`
* we enter the while loop
* we grab next frame in the loop and turn it to grayscale
* we calculate the optical flow with Farnerbach `flow = cv2.calcOpticalFlowFarneback(prevImg,nextImg,None,0.5,3,15,3,5,1.2,0)` passing default params
* the flow object contains vector flow cartesian info (x,y)
* we want to convert this into polar coordinates to magnitude and angle, when we get tis info we will map it to the HSV color mapping. magnitude will represent saturation and angle the hue
* if all moves in the same direction it will be colored the same way
* we conver to polar coordinates `mag, ang = cv2.cartToPolar(flow[:,:,0],flow[:,:,1],angleInDegrees=True)` angle in degrees
* i set the hue in hsv as the angle/2 to reduce the num of hues `hsv_mask[:,:,0] = ang/2`
* we set value channel in mask to the mag in 0-255 range `hsv_mask[:,:,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)`
* we convert the mask to bgr to be presentable `bgr = cv2.cvtColor(hsv_mask,cv2.COLOR_HSV2BGR)`
* we imshow the mask `bgr = cv2.cvtColor(hsv_mask,cv2.COLOR_HSV2BGR)`
* we add escape logic, renew the frame `prevIng = nextImg`
* and cleanup out of the  loop

### Lecture 61 - MeanShift and CAMShift Tracking Theory

* Some of the most basic tracking methods are MeanShift and CAMShift
* We ll first describe the general MeanShift algortithm, then learn how to apply it for image tracking
* Afterwards we will learn how to extend the MeanShift into CAMShift (Continuously Adaptive MeanShift)
* Imagine we have a set of x,y points and we want to assign them into clusters
* we will take all our data points and stack red and blue points on them (blue on top of red)
* the direction to the closest cluster centroid is determined by where most of the points nearby are at (weighted mean)
* so in each iteration each blue point will mov closer to where the most points are at, which is or will lead to the cluster center
* blue and red datapoints overlap completely in teh first iteration before the Meanshift algorithm starts
* at the end of iteration one, all the blue points move towards the clusters. 
* in our example in 3rd iteration all clusters reach convergence. there is no reason for more iterations as cluster means stop moving
* Meanshift algo wont always detect what may appear to us more reasonable
* In K-means algorithm (Machine Learning) we choose how many clusters we have beforehand
* How MeanShift applies to object tracking:
* meanshift can be given a target to track, calculate the color histogram of the target area, and then keep sliding the tracking window to the closest match (the cluster center)
* Just using Meanshift won't change the window size if the target moves away or towards the camera.
* We can use CAMShift to update the size of the Window

### Lecture 62 - MeanShift and CAMShift Tracking with OpenCV

* we start notebook and do the imports
* we start video capture
* we grab a frame
* we will get our ROI doing facetracking once in the first frame only
* we get the haarcascade obj in object `face_cascade = cv2.CascadeClassifier('../DATA/haarcascades/haarcascade_frontalface_default.xml')`
* we get the rects where face is dtected using cascades `face_rects = face_cascade.detectMultiScale(frame)`
* we grab the first face as we want to track only one `(face_x,face_y,w,h) = tuple(face_rects[0])` we convert it to tuple as it is needed for the algo
* we name our tuple tracking window `track_window = (face_x,face_y,w,h)`
* we set an ROI for tracking `roi = frame[face_y:face_y+h,face_x:face_x+w]`
* we use hsv colormapping `hsv_roi = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)`
* we will get the hsv histogram to backproject each frame in order to calculat ethe meanshift `roi_hist = cv2.calcHist(hsv_roi,[0],None,[180],[0,180])` we get hist for hue channel for vals 0-180
* the algo works with 0-255 so we normalize `cv2.normalize(roi_hist,0,255,cv2.NORM_MINMAX)`
* we set the termination criteria `term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,1)` for 10 iterations or eps=1
* we start our while loop
* if we have a frame we convert it to hsv
* we will calculate the backprojection based on the roi hist we have `dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)` note we work on 0-180
* using meanShift we get a new trackwindow pasing the previous one, the backpropagation and the term criteria `ret, track_window = cv2.meanShift(dst,track_window,term_citeria)`
* we will now draw a new rect on the image based on the new track window (we do tuple unpacking to get coords `x,y,w,h = track_window`)
* we add the rectangle `img2 = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),5)` and show the frame
* we add escape logic and cleanup
* we fix of the rect not resizing based on head size change using the CAMShift
```
        ret,track_window = cv2.CamShift(dst,track_window,term_criteria)
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv2.polylines(frame,[pts],True,(0,0,255),5)
```
* essentialy we draw a polyline and use CAMshift

### Lecture 63 - Overview of various Tracking API Methods

* [Boosting Tracker](http://www.vision.ee.ethz.ch/boostingTrackers/onlineBoosting.htm)
* [MIL Tracker](http://faculty.ucmerced.edu/mhyang/papers/cvpr09a.pdf)
* [KCF Tracker](https://arxiv.org/abs/1404.7584)
* [TDL Tracker](https://ieeexplore.ieee.org/document/6104061)
* [Median Flow Tracker](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.231.4285&rep=rep1&type=pdf)
* There a re many Object Tracking methods
* Fortunately, many have been designed as simple API calls with openCV
* We ll explore a few of these easy to use Object Tracking APIs and in next lect we ll use them with OpenCV
* BOOSTING TRACKER:
	* based off AdaBoost algorithm (the same underlying algorithm that the HAAR Cascade based Face Detection used)
	* Evaluation occurs across multiple frames
	* Pros: very well known and studied algorithm
	* Cons: Does not know when tracking has failed, there are many better techniques
* MIL TRACKER:
	* Multiple Instance Learning
	* Similar to BOOSTING, but considers a neighborhood of points around the current location to create multiple instances
	* Check the project page for details
	* Pros: good performance and does not drift as much as BOOSTING
	* Cons: failure to track an object may not be reported back, cannot recover from full obstruction
* KCF TRACKER:
	* Kernelized Correlation Filters
	* Exploits some properties of the MIL Tracker and the fact that many data points will overlap, leading to more accurate and faster tracking
	* Pros: better than MIL and BOOSTING, Great First Choice
	* Cons: Cnnot recover from full obstruction of object
* TDLD TRACKER:
	* Tracking, learning and Detection
	* The Tracker follows the object from frame to frame
	* The Detector localized all appearances that have been observed so far and corrects the tracker if necessary
	* The learning estimates detector's errors and updates it to avoid these errors in the future
	Pros: Good at tracking even with obstruction in frames, tracks well under large changes in scale
	* Cons: Can provide many false positives
* Median Flow Tracker:
	* Internally, this tracker tracks the object in both forward and backward directions in time and measures the discrepancies between these two trajectories
	* Pros: very good at reporting failed tracking, works well with predictable motion
	* Cons: Fails under large motion (fast moving objects)

### Lecture 64 - Tracking APIs with OpenCV

* we will use the course notebook
* we can select tracker at runtime
* we draw manually our roi using 'cv2.selectRoi(frame,False)'
* we use the roi to initialize tracker `ret = tracker.init(frame,roi)`
* in the loop we update tracker with new frames `success, roi = tracker.update(frame)`

## Section 8 - Deep Learning for Computer Vision

### Lecture  65 - Introduction to Deep Learning for Computer Vision

* Section Topics and Goals
	* High level overview of Machine Learning
	* Overview od understanding classification metrics
	* Cover Deep Learning Basics
	* Keras Basics
	* MNIST Data Overview
	* CNN Theory
	* Keras CNN
	* Deep Learning on Custom Image Files
	* Understanding YOLO v3
	* YOLO v3 with Python

### Lecture 66 - Machine Learning Basics

* Before diving into Deep Learning, lets work on understanding the general machine learning process we will be using
* The specific case of machine learning we will be conducting is known as supervised learning
* Machine Learning is a method of data analysis that automates analytical model building
* using algorithms that iteratively learn from data, machine learning allows computers to find hidden insights without being explicitly programmed where to look
* Supervised learning algorithms are trained using labeled examples, such as an input where the  desired output is known
* E.g. a picture have a category label such as either a Dog or Cat
* The Learning algorithm receives a set of inputs along with the corresponding correct outputs and the algorithm learns by comparing its actual output with correct outputs to find errors
* It then modifies the model accordingly
* Supervised learning is used in apps where hist data predicts futrure events
* Data Acq -> Data Cleaning => repeat[model training and building -> Model testing w/test  data ] -> model deployents
* Image classification and recognition is a very common and widely applicable use of deep learning and machine learning with OpenCV and Keras
* We continue by learning how to evaluate a classification task

### Lecture 67 - Understanding Classification Metrics

* We learned that after our machine leerarning process is complete, we will use performance metrics to evaluate how our model did
* The key classification metrics we need to understand are:
	* accuracy
	* recall
	* precision
	* F1-score
* Typically in any classification task our model can only achieve 2 results
* either model was correct in its prediction
* or our model was incorrect in its prediction
* correct and incorrect expands in situations where we have multiple classes
* for the purposes of explainin the metrics lets imagine binary classification situation where we have 2 available classes
* in our example we will try to predict if an image is a dog or cat
* as we deal with supervised learn we fit/train a model on training data and then test model on testing data
* once we have the model predictions from the X_test data we compare them with the trye yvals (correct labels)
* we repeat test process for all images in our test data
* at the end we will have a count of correct matches and a count of incorrect matches.
* the key point to take is that in real world not all incorrect or correct matches hold equal value
* we could organize our predicted values vs the real values in a confusion matrix
* Accuracy: 
	* is the number of correct predictions made by the model / total number of predictions
	* is useful when target classes are well balanced 
	* its not a good choice with unbalanced classes
	* = 1 - error rate (misclassification rate)
* Recall: 
	* ability of a model to find all relevant cases within a dataset.
	* number of true positives / num of true positives + num of false negatives
* Precision:
	* ability of a classification model to identify only the relevant data points
	* num of true positives / num of true positives + num of false positives
* Recall vs Precision:
	* often we have a trade-off between then
	* while recall expresses the abilityto find all relevant instances in a dataset, precision expresses the proportion of the data points our model says was relevant vs what actually  was relevant
* F1-score: 
	* in cases where we want to find an optimal blend of precision and recall we can combine the two metrics using the F1 score
	* it is the harmonic mean of precision and recall (F1 =  2 * (precision * recall)/(precision + recall))
	* we use harm mean vs average it punishes extreme values
	* a classifier with precision 1 and recall 0 has average of 0.5 but F1 is 0
* Comfusion Matrix
	* FP (False Positive) Type I error (prediction postive VS condition negative)
	* FN (False Negative) Type II error (prediction negative VS condition positive)

### Lecture 68 - introduction to Deep Learning Topics

* We will cover
	* neurons
	* Neural networks
	* Cost Function
	* Gradient Descent and BackPropagation

### Lecture 69 - Understanding a Neuron

* We skip 69-72 (see PythoDSML and Tesorflow Courses notes)

### Lecture 73 - Keras Basics

* We ll learn how to create a machine learning model with keras
* we ll start with some data on currency bank notes
* some of these bank notes were forgeries and others were legit
* researchers created a dataset from these banknotes by taking images of the notes and then extracting various numerical features based on the wavelets of the images
* the dataset is not images.
* we are doing general machine learning using Keras
* when we learn about CNN then we can expand on Keras to feed in image data (pixel images) into a network
* we open a notebook anmd import
```
import numpy as np
from numpy import genfromtxt
```
* we import data from csv using genfromtxt numpy method seting the delimiter to comma `data = genfromtxt('../DATA/bank_note_data.txt',delimiter=',')`
* our data is a (1372,5) array where last column contains the classes 0. = forgery 1. = legit
* first we need to separate teh label from the actual features 
```
y = labels = data[:,4] 
X = features = data[:,:4]
``` 
* i will now have to split my data to the train and test sets. we use sklearn lib to do it
```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
```
* when we work with NNs its a good idea to standardize or scale the data.we use sklearn for that
```
from sklearn.preprocessing import MinMaxScaler
scaler_object = MinMaxScaler()
scaler_object.fit(X_train)
scaled_X_train = scaler_object.transform(X_train)
```
* when we scale we always fit on train data (unless is data leakage to the model)
* all scaled feat vals now are between 0 and 1
* we start building our keras model with importing our DNN model and our layers type 
```
from keras.models import Sequential
from keras.layers import Dense
```
* we create our model and add layers to it
```
model = Sequential()
model.add(Dense(4,input_dim=4,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
```
* we compile our model adding loss method, oprimizer, and metrics `model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])`
* its time to fit or train the model `model.fit(scaled_X_train,y_train,epochs=50,verbose=2) we set the num of epochs
* to get the predictions (y_test_pred)  `model.predict_classes(scaled_X_test)`
* to get the model metrics  `model.metrics_names`
* we import confusion matrix to get the report of metrics and print the metrics
```
from sklearn.metrics import confusion_matrix, classification_report
predictions = model.predict_classes(scaled_X_test)
confusion_matrix(y_test,predictions)
print(classification_report(y_test,predictions))
```
* to save the model for production `model.save('my_banknote_classification_model.h5')`
* to reuse the model we load it with keras
```
from keras.models import load_model
newmodel = load_model('my_banknote_classification_model.h5')
```

### Lecture 74 - MNIST Data Overview

* A classic data set in Deep Learning is the MNIST data set
* We ll quickly cover some basics about it since w ll be using similar data concepts quite frequently during this section of the course
* The data set is easy to accesss with Keras. it has 60k training images 10k test images
* it contains hand written single digits from 0 to 9
* a single digit image can be represented as a numpy array
* they are 28x28 and have 1 color channel normalized (0-1)
* our input tensor is 4D (60000,28,28,1) or (samples,x,y,channels) for color images the las dimension would be 3
* for labels we ll use hot encoding. instad of having labels like 'one' 'two' etc w ll have a single array for each image. the orgiginal labels are given as list of nums [5,4,5,...,7,2] we will convert them to on-hot encoding (easy to do with Keras)
* Hot encoding:
	* the label is represented based of the index position in the label array
	* the corresponding label will be 1 at the index location and zero elsewere
	* eg 4 will have this label array [0,0,0,0,1,0,0,0,0,0]
	* works well with sigmoid
* As a result labels for training data ends up being a 2D array (60000,10)

### Lecture 75 - Convolutional Neural Networks Overview - Part One

* we just created a NN for already defined features
* what if we have the raw image data?
* we need to learn about CNNs in order to effectively solve the problems that image data can present
* just like the simple perceptron, CNNs also have their origins in biological research
* Hubel and Wiesel studied the structure of the visual cortex in mammals winning a Nobel prize in 1981
* their research revealed that neurons in the visual cortex had a small local receptive field (they are looking ata a subset of the image a person is viewing). these sections overlap and create the larger image
* these neuron in the visual cortex are only activated when they detect certtain things (e.g a horizontal line,a black circle etc)
* this idea then inspired an ANN architecture that would become CNN
* This architecture was implemented by Yann LeCun in 1998
* THe LeNet-5 architecture was first used to classify the MNIST data set
* When we learn about CNNs we often see a diagram with subsampling/pooling and convolution layers generating feature maps from an image
* Topics:
	* Tensors
	* DNN vs CNN
	* Convolutions and Filters
	* Padding
	* pooling Layers
	* Review Dropout
* Tensors are N-Dimensional Arrays that we build up to as we increase the level of dimension:
	* Scalar -3
	* Vector - [3,4,5]
	* Matrix - [[3,4],[5,6],[7,8]]
	* Tensor - [[1,2],[3,4]],[[5,6],[7,8]]
* Tensors make it very conveninent to feed in sets of images into our model - (I,H,W,C)
	* I: Images
	* H: Height of Image in Pixels
	* W: Width of Image in Pixels
	* C: Color Channels: 1-Grayscale, 3-RGB
* Lets explore the difference between a Densely Connected Neural Network and a Convolutional Neural Network
* Recall that we've aleady been able to create DNNs with tf.estimator API
* In a DNN every neuron in one layer is connected to every neuron in next layer
* IN a CNN each unit is connected to a smaller number of nearby units in next layer inspired by biology that in visual cortex that we only look at local receptive fields
* Why CNN? MNIST dataset is 28x28=784 . most images are at least 256x256 = >56k. this leads to too many params unscalable to new images
* Convolutions also have a major advantage for image processing, where pixels nearby to each other are much more correlated to each other for image detection
* Each CNN layer looks at an increasingly larger part of the image
* Having units only connected to nearby units also aids in invariance
* CNN also helps with regularization by limiting the search of weights to the size of the convolution
* Lets explore how the convolutional neural network relates to image recognition
* We start with the input layer, the image itself
* Convolutional layers are only connected to pixels in their respective fields
* we run into a possible issue for edge neurons. there may not be an input there for them. we can fix this by adding a padding of 0x around the image
* Converting a DNN to CNN with 1-D convolution. we have only local connections to next layer. the weights of the connections work as filters (e.g for edge detection)
* our filters have a size (how many neuron take part) and a stride hoew many neuron to skip to the next group
* we can stack multiple filters (conceptually verticaaly) adding a dimension to our tensors
* Each filter detects a different filter
* we describe and visualize these sets of neurons as sets of blocks
* In 2D convolutions (images) our layers (tensor) have 3D (FxWxH) if we have color image we add a dimension
* subsections of theimage translate to sections of the tensor (layer)
* Convolutional fitlers are commonly visualized as a grid system (direclty as image processing with kernels)

### Lecture 76 - Convolutional Neural Networks Overview - Part Two

* we saw what convolutions are
* we ll see what subsampling (pooling) layers
* Pooling layers will subsample the input image, which reduces the memory use4 and computer load as well as reducing the number of params
* say we have a layer of pixels in our inpute image
* for our MNIST digits set, each pixel had a value representing darkness
* we create a 2x2 pool (or 3x3 or XxX) of pixels and ealuate the max val. only the max val makes it to the next layer
* the pooling layer removes a lot of info. even a 2x2 pooling kernel with stride of 2 removes 75% of the input data
* Another technique deployed by CNN is Dropout
* Dropout can be thought as a form of regularization to help prevent overfitting
* During training, units are randomly dropped, along with their connections
* This helps prevent units from co adapting too much
* We ll see some famous CNN architectures
	* LeNet-5 by Yann LeCun
	* AlexNet by Alex Krizhevsky et al.
	* GoogleNet by Szegedy at Google Research
	* ResNet by Kaiming He et al

### Lecture 77 - Keras Convolutional Neural Networks with MNIST

* we open a new network
* we import mnist dataset from keras `from keras.datasets import mnist`
* we load train and test data `(x_train,y_train),(x_test,y_test) = mnist.load_data()`
* we check shape of x_train `x_train.shape` is (60000,28,28) there is no color channel
* we import matplotlib and plot the first image `plt.imshow(x_train[0,:,:],cmap='gray')`
* the y_train is (60000,0) so esentyally a 1d array of nums 0-9
* we want to hot encode them as if we feed them like this then network will get confuses as if its a regression rpoblem
* to hot encode we import `from keras.utils.np_utils import to_categorical`
* we do the hotencoding
```
y_cat_test = to_categorical(y_test,10)
y_cat_train = to_categorical(y_train,10)
```
* `y_cat_test.shape` is (10000,10)
* our train data are not normalized. `x_train[0].max()` is 255 . we normalize it to be 0-1 in a way that we dont need sklearn
```
x_train = x_train / x_train.max()
x_test = x_test / x_test.max()
```
* i need to reshape the data to convert it to be fed to a general network. we add the color channel. `x_train = x_train.reshape(60000,28,28,1)` we do sam e for x_test
* we start building our model
```
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPool2D,Flatten
```
* we create the model
```
model = Sequential()
# Convolutional Layer
model.add(Conv2D(filters=32,kernel_size=(4,4),input_shape=(28,28,1), activation='relu'))
# Pooling Layer
model.add(MaxPool2D(pool_size=(2,2)))
# Flatten out 2D --> 1D - Prepare for DNN feed
model.add(Flatten())
# Dense Layer
model.add(Dense(128,activation='relu'))
# Output Layer - Classifier
model.add(Dense(10,activation='softmax'))
# Compile
model.compile(loss='categorical_crossentropy',
             optimizer='rmsprop',
             metrics=['accuracy'])
```
* we get the summary of model `model.summary()`
* we train our model `model.fit(x_train,y_cat_train,epochs=2)`
* we evaluate `model.evaluate(x_test,y_cat_test)`
* we build the reports
```
from sklearn.metrics import classification_report,confusion_matrix
predictions = model.predict_classes(x_test)
print(classification_report(predictions,y_test))
```

### Lecture 78 - Keras Convolution Neural Networks with CIFAR-10

* we import the dataset `from keras.datasets import cifar10`
* we load the dataset `(x_train,y_train),(x_test,y_test) = cifar10.load_data()`
* we chexk shape `x_train.shape` (50000, 32, 32, 3) and `x_train[0].max()` so is unscaled
* we normalize
```
x_train = x_train / x_train.max()
x_test = x_test / x_test.max()
```
* we check labels the y are normal integer category forms. we ll hot encode them
```
from keras.utils.np_utils import to_categorical
y_cat_test = to_categorical(y_test,10)
y_cat_train = to_categorical(y_train,10)
```
we we build our model
```
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPool2D,Flatten
model = Sequential()
model.add(Conv2D(32,kernel_size=(4,4),input_shape=(32,32,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(32,kernel_size=(4,4),input_shape=(32,32,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.compile(loss='categorical_crossentropy',
             optimizer='rmsprop',
             metrics=['accuracy'])
```
* we train for 2 epochs `model.fit(x_train,y_cat_train,verbose=1,epochs=2)`
* we load a pretrained model
```
from keras.models import load_model
new_model = load_model('../../Computer-Vision-with-Python/06-Deep-Learning-Computer-Vision/cifar_10epochs.h5')
```
* we evaluate both. 2 epochs 0.59 acc 10eopchs 0.64 acc

### Lecture 80 - Deep Learning on Custom Images - Part One

* in real world apps we will have to work with real images ray jpeg images
* we will use real images
* in the CATS_DOGS folder with data there are two folders 'train' and 'test' each with 'CAT' and 'DOG' subfolders
* this is the default way of adding classification data in keras. also hav eto put it in the kjupyter notebook folder
* we open notwebook and import cv2 and matplotlib
* we load a cat from train folder `cat4 = cv2.imread('CATS_DOGS/train/CAT/4.jpg')` we clor corect it and show it
* we do the same for a dog image
* we note that images have different shapes
* we need to prepair the data for the model
* keras has a method that read data and prepares aflow of batches to pass in the model `from keras.preprocessing.image import ImageDataGenerator`
* image_geenrator also creates variations of the images for a stronger model
* we create an image generator passin in a lot of alteration to the image as params
```
image_gen = ImageDataGenerator(rotation_range=30,
                              width_shift_range=0.1,
                              height_shift_range=0.1,
                              rescale=1/255,
                              shear_range=0.2,
                              zoom_range=0.2,
                              horizontal_flip=True,
                              fill_mode='nearest')
```
* we use it on the dog image `image_gen.random_transform(dog2)` and plot it. every time we run it we get a different modified version of dog2
* next we will create a lot of modified images from our train directory. we use `image_gen.flow_from_directory('CATS_DOGS/train')` which creates a constant feed of images (randomized) to the model
* it returns a DirecoryIterator object. it also found how many classes (num of folders)

### Lecture 81 - Deep Learning on Custom Images - Part Two

* We will instroduce some slightly different imports to reflect the most recent  changes to Keras lib.
* THese are just a few different imports: MaxPool2D => MaxPooling2D, Adding activation functions separately
* we import model and layers (keras v2.2 style)
```
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout,Flatten,Conv2D,MaxPooling2D
```
* we create the model and add layers
```
model = Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(150,150,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=64,kernel_size=(3,3),input_shape=(150,150,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=64,kernel_size=(3,3),input_shape=(150,150,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])
```
* we set input_shape=(150,150,3) we will add this to image generator  flow from idr params. so that we get fed with images resized to have a uniform size
* we add dropout layer to avoid overfitting
* we select batch size and the input shape (same like in model)
```
input_shape = (150,150,3)
batch_size = 16
```
* we create our generator for train and test
```
train_image_gen = image_gen.flow_from_directory('CATS_DOGS/train',
                                               target_size=input_shape[:2],
                                               batch_size = batch_size,
                                               class_mode='binary')
test_image_gen = image_gen.flow_from_directory('CATS_DOGS/test',
                                               target_size=input_shape[:2],
                                               batch_size = batch_size,
                                               class_mode='binary')
```
* generator objects are loaded with attributes `train_image_gen.class_indices` shows us which num belongs to which class
* we train our model using the generator 
```
results = model.fit_generator(train_image_gen,epochs=1,steps_per_epoch=150,
                             validation_data=test_mage_gen,validation_steps=12)
```
* we set steps per epoch to limit the size of the epoch. in our case 150 batches of 16
* we will also run our validation in same run with 12 steps of 16
* to ignore warnings
```
import warnings
warnings.filterwarnings('ignore')
```
* we have thre results so we can evaluate thte model
* we can see its accuracy in 1st epoch validation with `results.history['acc']`
* we load a pretrained model for 100 epochs
```
from keras.models import load_model
new_model = load_model('../../Computer-Vision-with-Python/06-Deep-Learning-Computer-Vision/cat_dog_100epochs.h5')
```
* we do prediction using the pretrained model
	* we get an image path `dog_file  = 'CATS_DOGS/test/DOG/10005.jpg'`
	* we import image preproc from keras `from keras.preprocessing import image`
	* we load image resizing it `dog_img = image.load_img(dog_file,target_size=(150,150))`
	* we conver it to array `dog_img = image.img_to_array(dog_img)`
	* we reshape the array so that keras thinks its a batch of 1 image `dog_img = np.expand_dims(dog_img,axis=0)`
	* shape now is (1,150,150,3)
	* we normalize it `dog_img = dog_img /255`
	* we do the prediction `new_model.predict_classes(dog_img)` it is correct id  give class 1 (dog)
	* how sure it was? `new_model.predict(dog_img)`

### Lecture 82 - Deep Learning and Convolutional Neural Networks Assessment

* 

### Lecture 84 - Introduction to YOLO v3

* Let's learn about the state of the art image detection algorithm known as YOLO (You Only Look Once)
* YOLO can view an image and draw bounding boxes over what it perceives as identified classes
* In this lecture we will use version 3 of the YOLO Object Detection Algo, which is improved in terms of accuracy and speed
* What makes YOLO different?
	* prior detection systems repurpose classifiers or localizers to perform detection
	* they apply the model to an image at multiple locations and scales. High scoring regions of the image are considered detections
	* YOLO uses a totally different approach. Wea pply a single neural network to the the full image. This network divides the image into regions and predicts bounding boxes and probabilities for each region. these bounding boxes are weighted by the predicted probabilities
* YOLO has several advantages over classifier-based systems
* It looks at the whole image at test time so its predictions are informed by global context in the image
* It also makes predictions with a single network evaluation unlike systems like R-CNN which require thousands for a single image. This makes it xtremely fast more than 1000x faster than R-CNN and 100x fastr that Fast R-CNN
* In next version we will load an already trained YOLO model and see how we can use it with either image or video data
* We've set up an easy to use notebook. we just have to download the model weights file

### Lecture 86 - YOLO v3 with Python

* Let's explore how to implement YOLO v3 with Python
* we ll be using an implementation of YOLO v3 that has been trained on the COCO dataset
* COCO dataset has 1.5million object instances with 80 different obj.categories
* will use a YOLO v3 pretrained model to explore its capabilities
* We need many many days and a high end computer to train such a model
* this model is extremely complex 200MB h5 file
* we will place the yolo.h5 in the DATA dir of the YOLO folder
* we will use a ready notebook with easy to call functions
* [COCO dataset](http://cocodataset.org/#home)
* [COCO paper](https://arxiv.org/pdf/1405.0312.pdf)
* [YOLO v3 paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
* we do the imports
```
import os
import time
import cv2
import numpy as np
from model.yolo_model import YOLO
```
* we do image processing to prepare the input image for the model (frame or image we provide)
```
def process_image(img):
    """Resize, reduce and expand image.

    # Argument:
        img: original image.

    # Returns
        image: ndarray(64, 64, 3), processed image.
    """
    image = cv2.resize(img, (416, 416),
                       interpolation=cv2.INTER_CUBIC)
    image = np.array(image, dtype='float32')
    image /= 255.
    image = np.expand_dims(image, axis=0)

    return image
```
* we get the classes from a text file (that we provide)
```
def get_classes(file):
    """Get classes name.

    # Argument:
        file: classes name for database.

    # Returns
        class_names: List, classes name.

    """
    with open(file) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    return class_names
```
* we have the draw function that will fraw on the picture based on teh model outputs
```
def draw(image, boxes, scores, classes, all_classes):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box

        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(all_classes[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 1,
                    cv2.LINE_AA)

        print('class: {0}, score: {1:.2f}'.format(all_classes[cl], score))
        print('box coordinate x,y,w,h: {0}'.format(box))

    print()
```
* we have two methods one for images and one for video. the take the image the model the classes and return or draw the results
```
def detect_image(image, yolo, all_classes):
    """Use yolo v3 to detect images.

    # Argument:
        image: original image.
        yolo: YOLO, yolo model.
        all_classes: all classes name.

    # Returns:
        image: processed image.
    """
    pimage = process_image(image)

    start = time.time()
    boxes, classes, scores = yolo.predict(pimage, image.shape)
    end = time.time()

    print('time: {0:.2f}s'.format(end - start))

    if boxes is not None:
        draw(image, boxes, scores, classes, all_classes)

    return image
```
* we load the model and the classes
```
yolo = YOLO(0.6, 0.5)
file = 'data/coco_classes.txt'
all_classes = get_classes(file)
```
* this runs '/model/yolo.py' which runs the model
* the params 0.6 is obj threshold and 0.5 is nms thresh
* lower threshold is more detections but also more prone to errors
* code for detecting in image 
```
f = 'person.jpg'
path = 'images/'+f
image = cv2.imread(path)
image = detect_image(image, yolo, all_classes)
cv2.imwrite('images/res/' + f, image)
```

## Section 9 - Capstone Project

### Lecture 87 - Introduction to Capstone Project

* We will be creating a program that can detect a hand, segment the hand and count the number of fingers being held up

### Lecture 88 - Capstone Part One - Variables and Background function

* first we will define some global variables
* after. we will setup a function that updates a running average of the background values in an ROI
* This will later on allow us to detect new objects (hand) in the ROI
* in an empty frame we draw a ROI. we wait 60sec for the avg of the background in the roi ti be calculated. 
* then we enter our hand. it can be detected by the change in the backgrouns.
* we aply thresholding
* Strategy for counting fingers
	* grab the ROI
	* calculate a running average background val for 60 frames of video
	* once the avg value s found then the hand can enter the ROI
	once the hand enters the ROI, we will ise a convex hull to draw a polygon around the hand
	* we ll then calculate the center of the hand
	* then using math we will calculate the center of the hand against the angle of outer points to infer the finger count
* we start a new notebook
* we do our imports
```
import cv2
import  numpy as np
from sklearn.metrics import pairwise
```
* we create our global variables
```
background = None
accumulated_weight = 0.5
roi_top = 20
roi_bottom = 300
roi_right = 300
roi_left = 600
```
* our roi is preset
* we add a function to calculate the background value in the roi
```
def calc_accum_avg(frame,accumulated_weight):
    global background
    if (background is None):
        background = frame.copy()
        return None
    cv2.accumulateWeighted(frame,background,accumulated_weight)
```
* this method calculates the accumulated weight using a running average of passed frame (and the background).it updtates the global accumulated weight. the first time it sets the background equal to the frame

### Lecture 89 - Capstone Part Two - Segmentation

* 