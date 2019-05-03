import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def nothing(x):
    pass


def __main__():
	cap = cv.VideoCapture(0)


	fgbg = cv.bgsegm.createBackgroundSubtractorMOG()
	cv.namedWindow('edges')

	cv.createTrackbar('maxVal','edges',0,255, nothing)
	cv.createTrackbar('minVal','edges',0,255, nothing)
	
	while(True):

		ret,frame = cap.read()

    	# Our operations on the frame come here
		
		gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
		
		cv.imshow('gray',gray)

		filtered = cv.bilateralFilter(gray,9,75,75)

		cv.imshow('filtered',filtered)
		fgmask = fgbg.apply(frame)
		
		cv.imshow('subtracted',fgmask)

		maxVal = cv.getTrackbarPos('maxVal','edges')
		minVal = cv.getTrackbarPos('minVal','edges')

		if maxVal <  minVal:
			minVal = maxVal
		ret, thresh = cv.threshold(fgmask, minVal, maxVal, 0)
		cv.imshow('thresh',thresh)

		
		

		edges = cv.Canny(thresh,minVal,maxVal)
		cv.imshow('edges',edges)


		contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
		#print (len(contours[0]))
		if len(contours) > 0:
		#	cv.drawContours(frame, contours,0, (0,255,0), 3)
		#	print (len(contours[0]))
		#if len(contours) > 1:
		#	cv.drawContours(frame, contours,1, (0,0,255), 3)
		#if len(contours) > 25:
			cv.drawContours(frame, contours,len(contours) -1, (255,0, 0), 3)
		
		cv.imshow('contour',frame)


		#print (hierarchy)
		if cv.waitKey(1) & 0xFF == ord('q'):
			break

	# When everything done, release the capture
	cap.release()
	cv.destroyAllWindows()


if __name__ == "__main__":
	__main__()