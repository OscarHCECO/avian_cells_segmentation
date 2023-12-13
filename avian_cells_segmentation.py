def plot_comparison(image1,title1,image2,title2,image3,title3,image4,title4,image5,title5,image6,title6):

    fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(nrows=2,ncols=3, figsize=(8, 4))
    ax1.imshow(image1)
    ax1.set_title(title1)
    ax1.axis('off')
    ax2.imshow(image2)
    ax2.set_title(title2)
    ax2.axis('off')
    ax3.imshow(image3)
    ax3.set_title(title3)
    ax3.axis('off')
    ax4.imshow(image4)
    ax4.set_title(title4)
    ax4.axis('off')
    ax5.imshow(image5)
    ax5.set_title(title5)
    ax5.axis('off')
    ax6.imshow(image6)
    ax6.set_title(title6)
    ax6.axis('off')
#cell segmentation 
#libraries
import cv2
import numpy as np
import skimage as skimage
import matplotlib.pyplot as plt
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt
from skimage.exposure import histogram
from skimage.feature import canny
from scipy import ndimage as ndi
#load the images
imageGBR = cv2.imread('all (12).tif', cv2.IMREAD_COLOR)
imageRGB = cv2.cvtColor(imageGBR,cv2.COLOR_BGR2RGB)#to rgb
image8b = cv2.cvtColor(imageRGB, cv2.COLOR_BGR2GRAY)#to gray

#enhance contrast
clahe = cv2.createCLAHE(clipLimit=0.2)# i.e histogram equalization 
enhanced_8b = clahe.apply(image8b)#enhance contrast
ret, thresholded_gray = cv2.threshold(enhanced_8b, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)#otsu thresholding detect the nucleus
#we need to detect background and sure front ground (nuclei)
gray_dtransformed = cv2.distanceTransform(thresholded_gray, cv2.DIST_HUBER, 0)
ret, sure_fg = cv2.threshold(gray_dtransformed,0.1*gray_dtransformed.max(),255,0)
sure_fg = np.uint8(sure_fg)#change format
#sure_fg shows most of the stained nuclei and has less noise than thresholded
plot_comparison(imageGBR,"GBR",imageRGB,"RGB",enhanced_8b,"8b",thresholded_gray,"Thres",gray_dtransformed,"distance.2",sure_fg,"nuclei")
plt.show()

#to identify sure background we need to separate cytoplasm from background
#plt.imshow(enhanced_8b)
#plt.show()

thresholded_image = np.where(enhanced_8b > 200, 0, 255)#apply a threshold manually defined by plotting enhanced_8b, this value can vary!
#200 for image 10 and 12
#120 for image 8
thresholded_image=np.uint8(thresholded_image)#thresholded_image is sure background
_, sure_bg = cv2.threshold(thresholded_image,0,200,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)#otsu thresholding detect the nucleus
unknown = cv2.subtract(sure_bg,sure_fg)
ret, markers0 = cv2.connectedComponents(sure_fg)#markers based on sure_fg
markers0[unknown==200] = 15#add contrast to the cytoplasm
#first try to find contours 
markers1 = np.uint8(markers0)#change format
contours1, _ = cv2.findContours(markers1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_img = np.zeros_like(enhanced_8b)  # Create a blank image of the same size
cv2.drawContours(contour_img, contours1, -1, (255, 255, 255), 2)  # Draw all contours
plot_comparison(imageRGB,"RGB",enhanced_8b,"8b",sure_fg,"Sure foreground",
                sure_bg,"sure background", markers0,"markers ",contour_img,"contours")#,sure_fg,"Thresh.2",sure_bg1,"dilated fg",)
plt.show()
#contours are not good in separating each cell
#many cells are not being detected
