import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import time
import imutils
from cv2.cv2 import imshow, waitKey

img = cv.imread('sunflowers.jpg',0)
img2 = img.copy()
template = cv.imread('sunflowerstemplate.jpg',0)
w, h = template.shape[::-1]


methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

#gray = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

max_val_global = 0
for scale in np.linspace(0.2, 1.6, 41):
    
    start = time.time()
    
    img = img2.copy()
    resized = imutils.resize(img, width = int(img.shape[1] * scale))
    
    if resized.shape[0] < h or resized.shape[1] < w:
            break
    
    res = cv.matchTemplate(resized,template,cv.TM_CCOEFF_NORMED)
    
    min_val, max_val_local, min_loc, max_loc = cv.minMaxLoc(res)
    
    if max_val_local>max_val_global:
        max_val_global = max_val_local
        top_left = max_loc
        final_image = img
        final_scale = scale
        ret, final_res_thres = cv.threshold(res,0.5,1,cv.THRESH_BINARY_INV)
        final_res = res
        
    
bottom_right = (top_left[0] + w, top_left[1] + h)
cv.rectangle(final_image,top_left, bottom_right, 255, 2)
plt.subplot(221),plt.imshow(final_res,cmap = 'gray')
plt.title('Matching Result ' + str(final_scale)), plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(final_image,cmap = 'gray')
plt.title('Detected Point ' + str(max_val_global)), plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(final_res_thres,cmap = 'gray')
plt.title('Thresholded'), plt.xticks([]), plt.yticks([])
end = time.time()
plt.suptitle('cv.TM_CCORR_NORMED ' + str(end - start))
plt.show()