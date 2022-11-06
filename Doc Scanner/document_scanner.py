import cv2 as cv
import imutils as im
from corner_points import four_point_transform
from skimage.filters import threshold_local

# load the image and compute the ratio of the old height to the new height, clone it, and then resize it
image = cv.imread('img1.jpg')
ratio = image.shape[0] / 500.0
orig = image.copy()
image = im.resize(image, height = 500)
orig_resized=im.resize(orig, height=500)

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) #for img= threshold values are 75, 200
gray = cv.GaussianBlur(gray, (5, 5), 0)      #for img1= 43, 200
edged = cv.Canny(gray, 119, 10)              
# kernel = np.ones((3, 3))
# imgDial = cv.dilate(edged, kernel, iterations=2) # APPLY DILATION
# edged = cv.erode(imgDial, kernel, iterations=1)  # APPLY EROSION

# find the contours in the edged image
cnts = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
cnts = im.grab_contours(cnts)
cnts = sorted(cnts, key = cv.contourArea, reverse = True) #sort the contours according to size in descending order
# loop over the contours
for c in cnts:
	# approximate the contour
	peri = cv.arcLength(c, True)
	approx = cv.approxPolyDP(c, 0.02 * peri, True)
	# if our approximated contour has four points, then we can assume that we have found our page size
	if len(approx) == 4:
		screenCnt = approx
		break
cv.drawContours(image, [screenCnt], -1, (0, 255, 0), 2) #show the contour (outline) of the piece of paper
# apply the four point transform to obtain a top-down view of the original image
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
# convert the warped image to grayscale, then threshold it

warped = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
T = threshold_local(warped,11, offset = 7, method = "gaussian") # to give it that 'black and white' paper effect
warped = (warped > T).astype("uint8") * 255

cv.imshow("Scanned", im.resize(warped, height = 650))

# cv.imshow("Outline", image)
cv.imshow("Image", orig_resized)
# cv.imshow("Edged", edged)

cv.waitKey(0)

