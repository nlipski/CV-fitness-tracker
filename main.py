import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt
from skimage.measure import compare_ssim

import time

# Raspberry Pi Camera packages
from picamera.array import PiRGBArray
from picamera import PiCamera


# Global variables
WIDTH = 640
HEIGHT = 480
FRAMERATE = 32
FRAMES_TO_SKIP = 60

def nothing(x):
		pass

def prep_image(frame):
	
	result = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	#result = cv2.bilateralFilter(result,9,75,75)
	result = cv2.GaussianBlur(result,(5,5),0)
	# use Gaussian or bilaterak filter for smoothing the image 
	# Gaussian is faster, but it doesn't preserve edges as well as bilateral
	return result

def detect_edges(frame, minVal, maxVal):

	frame = cv2.medianBlur(frame, 5)
	result = cv2.Canny(frame, minVal, maxVal)
	# edges below minVal are rejected,
	# the ones above maxVal are added, 
	# and the ones are between are added only if they're adjusted to strong adges
	return result

def process_image(cap):

	fgbg = cv2.createBackgroundSubtractorMOG2()
	count = 0

	rawCapture = PiRGBArray(cap, size=(WIDTH, HEIGHT))
	# Let camera warm up
	time.sleep(0.1)

	print("Beginning to take background image")
	for frame in cap.capture_continuous(rawCapture, format="bgr", use_video_port=True):

		orig_frame = frame.array
		if orig_frame == None:
			return None

		# skip first 60 frames to make sure nothing is moving  
		if count < FRAMES_TO_SKIP:
			count += 1
			continue
		print("Camera warm up complete")
		mod_frame = prep_image(orig_frame)
		fgmask = fgbg.apply(mod_frame)
		
		rawCapture.truncate(0)

		# diff to see that nothing has been moving  
		if (cv2.countNonZero(fgmask)) == 0:
			print("Background capture is complete")
			return orig_frame

		print("Something is moving! Can't take background image.\nRepeating the process. ")

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

def process_stream(cap, background):
	
	cv2.namedWindow('edges')
	cv2.createTrackbar('maxVal','edges',180,255, nothing)
	cv2.createTrackbar('minVal','edges',30,255, nothing)
	cv2.createTrackbar('thresh','edges',1,255, nothing)
	
	kernel = np.ones((2,2),np.uint8)

	rawCapture = PiRGBArray(cap, size=(WIDTH, HEIGHT))
	# Let camera warm up
	time.sleep(0.1)

	print("Beginning to capture the stream")

	for frame in cap.capture_continuous(rawCapture, format="bgr", use_video_port=True):

		orig_frame = frame.array
		if orig_frame == None:
			break

		maxVal = cv2.getTrackbarPos('maxVal','edges')
		minVal = cv2.getTrackbarPos('minVal','edges')
		thresh = cv2.getTrackbarPos('thresh','edges')

		if maxVal <  minVal:
			minVal = maxVal
			# need to modify trackbars
			# use setters instead and move outside of loop

		mod_frame = background_extraction(orig_frame, background,thresh)
		closing = cv2.morphologyEx(mod_frame, cv2.MORPH_OPEN, kernel)
		
		contours,__ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		result = np.zeros_like(frame)
		cutout = np.zeros_like(frame)
		
		if len(contours) != 0:
			max_contour = contours[0]
			for contour in contours:
				if cv2.contourArea(contour) > cv2.contourArea(max_contour):
					max_contour = contour

			contours_poly = cv2.approxPolyDP(max_contour, 3, True)
			boundRect = cv2.boundingRect(contours_poly)
			moment = cv2.moments(max_contour)
			
			if cv2.contourArea(max_contour) >= 3000:
				print ("stream: Found contour with area: " + str(cv2.contourArea(max_contour)))
				cv2.drawContours(result, max_contour , -1, (255,0,255))
				cv2.rectangle(result, (int(boundRect[0]), int(boundRect[1])),(int(boundRect[0]+boundRect[2]), int(boundRect[1]+boundRect[3])), (70,150,50), 2)

		cv2.imshow('result', result)
		cv2.imshow('extracted', closing)

		rawCapture.truncate(0)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

def background_extraction(frame, background, thresh):
	frame = prep_image(frame)
	background = prep_image(background)

	ret, result = compare_ssim(frame, background, full=True)
	result = (result * 255).astype("uint8")

	ret,thresh_img = cv2.threshold(result, thresh, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

	return thresh_img 


def capture_init():

	camera = PiCamera()
	if camera == None:
		return None
	camera.resolution = (WIDTH, HEIGHT)
	camera.framerate = FRAMERATE

	return camera

def capture_finit(cap):
	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()

def main():
	
	cap = capture_init()

	if cap == None:
		return 

	background = process_image(cap)
	cv2.imwrite("background.jpg", background)
	process_stream(cap, background)
	capture_finit(cap)


if __name__ == "__main__":
	
	try:
		main()
	except:
		raise()