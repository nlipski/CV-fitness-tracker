import cv2
import numpy as np
from matplotlib import pyplot as plt

def detect_edges(frame, minVal, maxVal):
	
	result = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	#result = cv2.bilateralFilter(result,9,75,75)
	result = cv2.GaussianBlur(result,(5,5),0)
	# use Gaussian or bilaterak filter for smoothing the image 
	# Gaussian is faster, but it doesn't preserve edges as well as bilateral

	result = cv2.Canny(result,minVal,maxVal)
	# edges below minVal are rejected,
	# the ones above maxVal are added, 
	# and the ones are between are added only if they're adjusted to strong adges
	return result


def process_stream(cap):
	
	cv2.namedWindow('edges')
	cv2.createTrackbar('maxVal','edges',0,255, nothing)
	cv2.createTrackbar('minVal','edges',0,255, nothing)

	while(True):

		ret,frame = cap.read()
		if ret == 0:
			break

		maxVal = cv2.getTrackbarPos('maxVal','edges')
		minVal = cv2.getTrackbarPos('minVal','edges')
		
		if maxVal <  minVal:
			minVal = maxVal

		#ret, thresh = cv2.threshold(fgmask, minVal, maxVal, 0)

		edges = detect_edges(frame, minVal, maxVal)

		cv2.imshow('edges', edges)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

def capture_init(width, height):
	
	cap = cv2.VideoCapture(0)
	if cap == None:
		return 
	cap.set(3,width) #width=640
	cap.set(4,height) #height=480

	return cap

def main():
	
	cap = capture_init(640,480)

	if cap == None:
		return 

	process_stream(cap)

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	
	try:
		main()
	except:
		raise()