import cv2
import imutils
import numpy as np
import os
import tensorflow as tf 
import argparse

def process(image):
	'''
	Applies Gaussian Blur to image to reduce the noise and enable better and faster processing.
	Applies Adaptive Threshold to image on the blurred image to convert it into a binary image. All lines are highlighted in white, rest everything is black.
	
	INPUT - image - Greyed image
	OUTPUT - Processed image
	'''
	
	#Applying gaussian blur
	blur = cv2.GaussianBlur(image, (9,9), 0)
	#show_image("Blurred", blur)

	#Applying adaptive Threshold
	adthresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
	#show_image("Adaptive Threshold", adthresh)
	
	return adthresh

def find_puzzle(image):
	'''
	This function finds the biggest contour with four corners (the one with max area) and returns it. The program assumes that the biggest four cornered contour in the image would be the puzzle box.
	
	INPUT - image - processes image(Image should be thresholded so as to find contours more efficiently
	OUTPUT - The biggest contour in the image
	'''
	
	#Finding contours
	cnts, heirarchy = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
	#Selecting the biggest contour based on area
	biggest = None
	max_area = 0
	for c in cnts:
		a = cv2.contourArea(c)							#Get contour area
		peri = cv2.arcLength(c, True)					#Get contour perimeter
		app = cv2.approxPolyDP(c, 0.02 * peri, True)	#Get all corners in the contour
		if a > max_area and len(app) == 4:				
			max_area, biggest = a, app
	
	return biggest

def get_eagle_view(image, biggest):
	'''
	This function applies Warps perspective transform to the greyed image. It crops the biggest contour (puzzle box) from the image and straightens the image to get the top view of the image
	
	INPUT - image - greyed image to apply the warp perspective transform to
			biggest - the contour to be cropped and straightened
	OUTPUT - returns the cropped and straightened image of puzzle box
	'''
	
	#Changing the biggest array to apply persective transformation. Reshaping it from 3D to 2D array
	biggest = np.reshape(biggest, (4,2))
	#Creating a numpy array of zeros of shape (4,2) same as biggest. We will reorder biggest using new. In the end, new = sorted biggest.
	#Sorting is as follows:
	#new[0] = top-left corner
	#new[1] = bottom-left corner
	#new[2] = bottom-right corner
	#new[3] = top-right corner
	new = np.zeros((4,2), np.float32)
	#Adds the elements in each row and stores it in add. Ex: if biggest = [[2,3],[3,4],[1,5],[5,3]], then add = [5,7,6,8]
	add = np.sum(biggest, axis=1)
	#Assigns the row with minimum sum of biggest to new[0]. argmin() returns the index of smallest element. That index is supplied to biggest.
	new[0] = biggest[np.argmin(add)]
	#Assigns the row with maximum sum of biggest to new[2]. argmax() returns the index of largest element. That index is supplied to biggest.
	new[2] = biggest[np.argmax(add)]
	#Calculates difference of the elements in each row and stores it in diff. Ex: if biggest = [[2,3],[3,5],[1,5],[5,3]], then diff = [-1,-2,-4,2]
	diff = np.diff(biggest, axis=1)
	#Assigns the row with minimum diff of biggest to new[1]. argmin() returns the index of smallest element. That index is supplied to biggest.
	new[1] = biggest[np.argmin(diff)]
	#Assigns the row with maximum diff of biggest to new[3]. argmax() returns the index of largest element. That index is supplied to biggest.
	new[3] = biggest[np.argmax(diff)]
	
	#Creating a new numpy array called 'pts' with same ordering of points as 'new'. This will be used for perspective transform. The new image thst will be created will have the size (450,450)
	pts = np.float32([[0,0], [450,0], [450,450], [0,450]])

	#applying perspective transformation
	M = cv2.getPerspectiveTransform(new, pts)
	puzz = cv2.warpPerspective(image, M, (450,450))
	
	return puzz
	
def get_grid():
	'''
	Slices the puzzle box into 91 cells. It creates a list 'squares' that contains a tuple of top-left and bottom-right points of each cell as its entries
	
	INPUT - no input
	OUTPUT - list 'squares' 
	'''
	#Dividing image into grids(cells)
	squares = []
	#Since each edge if 450px, each cell should have an edge of (450/9)px, i.e., 50px
	side = 450/9
	for i in range(9):
		for j in range(9):
			#Calculating the top-left point
			top_left = (i*side, j*side)
			#Calculating the bottom-right point
			bottom_right = ((i+1)*side, (j+1)*side)
			squares.append((top_left, bottom_right))
	
	return squares

def draw_grid(puzz, squares):
	'''
	Draws the grid on the image based on entries in 'squares'. 'squares' containes the top-left and bottom-right points using which we can draw rectangles. The function draws rectangles for each entry in 'squares'
	
	INPUT - puzz - image on which the grid is to be drawn
	OUTPUT - squares - list containing the top-left and bottom-right points
	'''
	
	#Creating a copy of image so the original image is not destroyed
	out = puzz.copy()
	#Drawing rectangles
	for s in squares:
		cv2.rectangle(out, tuple(int(x) for x in s[0]), tuple(int(x) for x in s[1]), 255, 2)
	show_image("Grid", out)

def show_image(title, image):
	'''
	Displays the image and waits for key interrupt before resuming the program. Destroys the image window after key interrupt
	
	INPUT - title - string to be displayed on title bar of image window
			image - image to be displayed
	OUTPUT - no output
	'''
	
	#Display image
	cv2.imshow(title, image)
	#Wait for key interrupt
	cv2.waitKey(0)
	#Destroy the image window
	cv2.destroyAllWindows()
	
def process_cells(puzz, squares, size=28):
	'''
	Creates a list of the image of each cells(91 in total) and returns that list. Calls extract_digit() for each entry in squares.
	
	INPUT - puzz - image of puzzle
			squares - list containing entries of top-left and bottom-right points for each cell
			size - size of each image of cell (optional argument. By default, the value is 28)
	OUPUT - digits - list containing the images of each cell. Each image has a size of (28,28) px
	'''
	
	digits = []
	#Processing puzzle image
	puzz = process(puzz)
	
	for s in squares:
		digits.append(extract_digit(puzz, s, size))
	
	return digits

def largest_feature(i, tl, br):
	'''
	Retreives the largest connected feature in the image and returns the bounding points of that feature
	
	INPUT - i - cell image
			tl - list of [margin, margin] of cell image. Gives top-left point
			br - list of [w-margin, h-margin] of cell image. GIves bottom-right point
	OUTPUT - img - cell image
			 numpy array of box (top-left, bottom-right) points
			 coordinates - coordinates of largest connected feature in cell
	'''
	
	img = i.copy()
	h, w = img.shape[:2]
	max_area = 0
	coordinates = (None, None)
	
	if tl is None:
		tl = [0, 0]
	if br is None:
		br = [w, h]
	
	#Getting the largest connected feature using cv2.floodFill()
	for x in range(tl[0], br[0]):
		for y in range(tl[1], br[1]):
			if img.item(y,x) == 255 and x < w and y < h:
				area = cv2.floodFill(img, None, (x,y), 64)
				if area[0] > max_area:
					max_area = area[0]
					coordinates = (x, y)
	
	for x in range(w):
		for y in range(h):
			if img.item(y,x) == 255 and x < w and y < h:
				cv2.floodFill(img, None, (x,y), 64)
	
	#Creating cover on which largest connected feature will be pasted
	cover = np.zeros(((h + 2), (w + 2)), np.uint8)
	
	if all([p is not None for p in coordinates]):
		cv2.floodFill(img, cover, coordinates, 255)
	
	top, bottom, left, right = h, 0, w, 0
	
	for x in range(w):
		for y in range(h):
			if img.item(y, x) == 64:
				cv2.floodFill(img, cover, (x, y), 0)
			
			if img.item(y, x) == 255:
				top = min(y, top)
				bottom = max(y, bottom)
				left = min(x, left)
				right = max(x, right)
			
	box = [[left, top], [right, bottom]]
	return img, np.array(box, dtype='float32'), coordinates

def align(img, size, margin=0, background=0):
	'''
	Centres the digit in the cell image
	
	INPUT - img - cell image
			size - size of output image i.e., (28,28)
			margin - optional argument (By default, it is 0)
			background - optional argument (By default, it is 0)
	OUTPUT - the cell image with digit centred in it
	'''
	
	h, w = img.shape[:2]
	
	#Helper function to return centre of image
	def centre(length):
		if length % 2 == 0:
			side1 = int((size - length) / 2)
			side2 = side1
		else:
			side1 = int((size - length) / 2)
			side2 = side1 + 1
		return side1, side2
	
	if h > w:
		t_pad = int(margin / 2)
		b_pad = t_pad
		ratio = (size - margin) / h
		w, h = int(ratio *  w), int(ratio * h)
		l_pad, r_pad = centre(w)
	else:
		l_pad = int(margin / 2)
		r_pad = l_pad
		ratio = (size - margin) / w
		w, h = int(ratio * w), int(ratio * h)
		t_pad, b_pad = centre(h)
		
	img = cv2.resize(img, (w,h))
	img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None, background)
	return cv2.resize(img, (size, size))
	
def extract_digit(img, sq, size):
	'''
	Calls largest_feature() func on each cell image to check whether the cell contains a digit or is empty. Calls align() func on the cell image if a digit is found, else returns a numpy array of size (28,28) containing all zeros(black image).
	
	INPUT - img - the puzzle image
			sq - a tuple containing the top-left and bottom-right points of cell
			size - size of cell image
	OUTPUT - calls align() on cell image and size if digit is found in cell
			 returns a numpy array of size (28,28) if no digit is found in cell
	'''
	
	#Slice image to get a single cell
	dig = img[int(sq[0][1]):int(sq[1][1]), int(sq[0][0]):int(sq[1][0])]
	
	#Get height and width to calculate margin
	h, w = dig.shape[:2]
	margin = int(np.mean([h,w])/2.5)
	
	#Calling largest_feature() to get the bounding coordinates of digit in cell
	_, box, seed = largest_feature(dig, [margin, margin], [w - margin, h - margin])
	#Slicing the cell image even furthur
	dig = dig[int(box[0][1]):int(box[1][1]), int(box[0][0]):int(box[1][0])]
	
	#Calculate width and height of new image from coordinates returnes from largest_feature()
	w = box[1][0] - box[0][0]
	h = box[1][1] - box[0][1]
	
	#Checking if cell contains digit or not and calls align() funcn if present, else returns a numpy array of zeros
	if w > 0 and h > 0 and (w*h) > 100 and len(dig) > 0:
		return align(dig, size, 4)
	else:
		return np.zeros((size, size), np.uint8)

def show_digits(digits, colour=255):
	'''
	Creates an image containing all the cell images
	
	INPUT - digits - list containing cell images
			color - optional arguments (By default, it is 255)
	OUTPUT - no output
	'''
	
	rows = []
	with_border = [cv2.copyMakeBorder(img.copy(), 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, colour) for img in digits]
	for i in range(9):
		row = np.concatenate(with_border[i * 9:((i + 1) * 9)], axis=1)
		rows.append(row)
	
	img = np.concatenate(rows)
	show_image("", img)

def store_images_and_get_digits(digits, sudoku=[]):
	'''
	Stores all the cell images as an individual image in a folder 'digitsAll' mentioned in 'path_digitsAll' variable.
	Stores all the cell images as an individual image that contain any digit in a folder 'digitsAll' mentioned in 'path_digits' variable
	Uses a ML digit recognition model to recognize the digits in the cell images and stores it in a list called 'sudoku'. Calls digit_recognize() func
	
	INPUT - digits - list containing the cell images
			sudoku - optional argument (Used for initialization)
	OUTPUT - sudoku - 1D list containing the digits in puzzle
	'''
	
	path_digitsAll = "C:\\Users\Rohan\Desktop\digitsAll"
	path_digits = "C:\\Users\Rohan\Desktop\digits"
	
	#Storing all the images in 'digitsAll' folder
	os.chdir(path_digitsAll)
	n = 0
	for c in digits:
		filename = "image" + str(n) + ".jpg"
		cv2.imwrite(filename, c)				#Storing image
		n += 1

	os.chdir(path_digits)
	
	#Deleting all files in folder 'digits'
	for filename in os.listdir(path_digits):
		os.remove(filename)
	
	#Loading ML model for digit recognition
	model = tf.keras.models.load_model("C:\\Users\Rohan\Desktop\model\digit_recog_comp_font2.h5")
	
	#Loads image file of each digit from 'digitsAll' folder and counts the number of white pixels int it. If the number of pixels exceed 150 or is equal to 150, it is assumed that it contains a digit. Calling ML model on that image and storing it in 'digits' folder
	for i in range(81):
		filename = path_digitsAll + "\image" + str(i) + ".jpg"
		a = cv2.imread(filename, 0)
		res = cv2.countNonZero(a)					#Counting number of white pixels
		if res >= 150:
			name = "image" + str(i) + ".jpg"
			cv2.imwrite(name, a)
			num = digit_recognize(a, model)			#Call digit_recognize() to recognize the digit
			sudoku.append(num)
		else:
			sudoku.append(0)
	
	return sudoku

def digit_recognize(image, model):
	'''
	Uses the ML model to recognize the digit in image.
	
	INPUT - image - cell image
			model - ML model
	OUTPUT - the digit in the image
	'''
	
	#Resizing the image to (28,28) px
	h, w = image.shape[:2]
	r = 28/w
	dim = (28, int(r*h))
	image = cv2.resize(image, dim)
	
	#Changing all values in the image to either 255(white) or 0(black)
	for i in range(28):
		for j in range(28):
			if image[i][j] < 150:			#If pixel value exceeds 150, it must be part of digit
				image[i][j] = 255
			else:
				image[i][j] = 0
	
	#Rehaping image to match the input of model
	image = image.reshape((1, 784)).astype('float32')
	image = image / 255
	
	#Prediction of digit
	prediction = model.predict(image)
	return prediction.argmax()
	
def transpose(digits):
	'''
	Makes transpose of 2D list
	
	INPUT - digits - list to be transposed
	OUTPUT - dig - transposed list
	'''
	
	t = []
	index = 0
	for i in range(9):
		d = []
		for j in range(9):
			d.append(digits[index])
			index += 1
		t.append(d)
	dig = []
	for i in range(9):
		for j in range(9):
			temp = t[j][i]
			dig.append(temp)
	
	return dig

def make_puzzle(sudoku):
	'''
	Converts the 1D digit list to 2D digit list in the form of puzzle
	
	INPUT - sudoku - 1D digit list
	OUTPUT - puzz - 2D digit list
	'''
	puzz = []
	index = 0
	for i in range(9):
		temp = []
		for j in range(9):
			temp.append(sudoku[index])
			index += 1
		puzz.append(temp)
	
	return puzz

#Taking Arguments from command line (Here we need only one argument, i.e., the image)
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

#Reading the image given as argument
image = cv2.imread(args['image'])
#show_image("Image", image)

#Resizing the image to (300, 300) pixels. Faster processing on smaller images. We calculate the ratio before resizing to maintain aspect ratio
h, w, d = image.shape
r = 300/w
dim = (300, int(r*h))
image = cv2.resize(image, dim)

#Converting to grayscale
grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#show_image("Greyscale", grey)

#Calling process() on greyed image
adthresh = process(grey)

#Calling find_puzzle() on the processed image
biggest = find_puzzle(adthresh)
#Creating a copy of image so it doesn't get destroyed when we draw contours on original image
i = image.copy()
#Drawing biggest contour (contour that we received from find_puzzle()) on the image
cv2.drawContours(i, [biggest], -1, (0,0,255), 2)
#show_image("Contours", i)

#Calling get_eagle_view() on the greyed image and biggest contour
puzz = get_eagle_view(grey, biggest)
#show_image("Puzzle", puzz)

#Calling get_grid() func
squares = get_grid()
#Drawing the grid on image. Calling the draw_grid() func
#draw_grid(puzz, squares)

#Calling process_cells() on puzz image and squares
digits = process_cells(puzz, squares)

#Calling transpose() on 'digits' list
digits = transpose(digits)

#Calling show_digits func on 'digits' list
#show_digits(digits)

#Calling store_images_and_get_digits() on 'digits' list
sudoku = store_images_and_get_digits(digits)

#print(sudoku)

#Convert the 1D list to 2D list(rows and columns) to simulate a puzzle
puzzle = make_puzzle(sudoku)

#print(puzzle)

#Creating command to execute the C program to solve the puzzle
#C program takes the list of numbers as arguments
path = "C:\\Users\Rohan\Documents"
os.chdir(path)
sudoku = [str(i) for i in sudoku]
cmd = "solve-sudoku "
cmd += " ".join(sudoku)
#print("Python Script: ", cmd)
#Executing the C program from command line
os.system(cmd)
