import cv2
import numpy as np
from matplotlib import pyplot as plt

def nothing(x):
    pass

def get_background():

	cap = cv2.VideoCapture(0)

	ret,frame = cap.read()
	
	i = 0
	mean = frame
	temp = np.float32(frame)

	while (i < 200) :
		ret,frame = cap.read()
		cv2.accumulate(frame,mean)
		i += 1
	mean = mean /200.0
	return mean





def __main__():
	cap = cv2.VideoCapture(0)


	fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
	cv2.namedWindow('edges')

	cv2.createTrackbar('maxVal','edges',0,255, nothing)
	cv2.createTrackbar('minVal','edges',0,255, nothing)

	background = get_background()

	cv2.imshow('background',background)

	while(True):

		ret,frame = cap.read()

    	# Our operations on the frame come here
		
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		#cv.imshow('gray',gray)

		filtered = cv2.bilateralFilter(gray,9,75,75)

		#cv.imshow('filtered',filtered)
		fgmask = fgbg.apply(frame)
		
		#cv.imshow('subtracted',fgmask)

		maxVal = cv2.getTrackbarPos('maxVal','edges')
		minVal = cv2.getTrackbarPos('minVal','edges')

		if maxVal <  minVal:
			minVal = maxVal
		ret, thresh = cv2.threshold(fgmask, minVal, maxVal, 0)
		cv2.imshow('thresh',thresh)

		
		

		edges = cv2.Canny(thresh,minVal,maxVal)
		#cv.imshow('edges',edges)


		contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		#print (len(contours[0]))
		if len(contours) > 0:
		#	cv.drawContours(frame, contours,0, (0,255,0), 3)
		#	print (len(contours[0]))
		#if len(contours) > 1:
		#	cv.drawContours(frame, contours,1, (0,0,255), 3)
		#if len(contours) > 25:
			cv2.drawContours(frame, contours,len(contours) -1, (255,0, 0), 3)
		
		#cv.imshow('contour',frame)


		#print (hierarchy)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	__main__()