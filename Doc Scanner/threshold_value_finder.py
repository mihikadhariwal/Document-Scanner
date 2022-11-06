import cv2 as cv
import numpy as np
import imutils as im

def nothing(x):
    pass

cv.namedWindow("Trackbars")
cv.resizeWindow("Trackbars", 360, 240)
cv.createTrackbar("Threshold1", "Trackbars", 0,255, nothing)
cv.createTrackbar("Threshold2", "Trackbars", 0, 255, nothing)

while True:

    def valTrackbars():
        Threshold1 = cv.getTrackbarPos("Threshold1", "Trackbars")
        Threshold2 = cv.getTrackbarPos("Threshold2", "Trackbars")
        src = [Threshold1,Threshold2]
        return src

    valTrackbars()

    img=cv.imread('img2.jpg')
    ratio = img.shape[0] / 500.0
    img=im.resize(img, 600, 600)
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # CONVERT IMAGE TO GRAY SCALE
    imgBlur = cv.GaussianBlur(imgGray, (5, 5), 1) # ADD GAUSSIAN BLUR
    thres=valTrackbars() # GET TRACK BAR VALUES FOR THRESHOLDS
    imgThreshold = cv.Canny(imgBlur,thres[0],thres[1]) # APPLY CANNY BLUR
    kernel = np.ones((5, 5))
    imgDial = cv.dilate(imgThreshold, kernel, iterations=2) # APPLY DILATION
    imgThreshold = cv.erode(imgDial, kernel, iterations=1)  # APPLY EROSION

    
    cv.imshow('Canny', imgThreshold)
    cv.imshow('img', img)
    # cv.imshow("Contours", imgContours)
    # cv.imshow("scanned", dst)
    # cv.imshow("Original", im.resize(img, height = 650))
    # cv.imshow("Scanned", im.resize(warped, height = 650))
    
    if(cv.waitKey(1)==ord('x')):
        break
    