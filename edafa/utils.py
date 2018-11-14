import cv2
import math
import numpy as np


# EXTENSIONS = ['jpg','png','tif','tiff']
AUGS = ['NO','ROT90','ROT180','ROT270','FLIP_UD','FLIP_LR','BRIGHT','CONTRAST','GAUSSIAN', 'GAMMA']
MEANS = ['ARITH','GEO']

def cint(num):
	"""
	Returns ceiling of number as type integer
	:param num: input number

	:returns: integer ceil number
	"""
	return int(math.ceil(num))

def rint(num):
	"""
	Returns rounded number as type integer
	:param num: input number

	:returns: integer rounded number
	"""
	return int(round(num))

def get_max(bits):
	"""
	Returns max value represented by given number of bits
	:param bits: number of bits

	:returns: 2^(bits) - 1
	"""
	return 2**bits -1

def add_reflections(img,in_patch_size,out_patch_size):
	"""
	Add mirror reflections to the passed image

	:param img: input image
	:param in_patch_size: model input patch size
	:param out_patch_size: model output patch size (would be different in case of padding)

	:returns: image padded with reflections
	"""
	O_H = img.shape[0]
	O_W = img.shape[1]
	
	padding = rint((in_patch_size - out_patch_size)/2.)

	top = padding
	left = padding
	right = rint(in_patch_size - (O_W % in_patch_size)) + padding
	bottom = rint(in_patch_size - (O_H % in_patch_size)) + padding

	# TODO: double check this condition
	if top ==0 or left ==0 or right ==0 or bottom==0:
		return img

	return cv2.copyMakeBorder(img,
							top, 
							bottom,
							left, 
							right,
							cv2.BORDER_REFLECT_101)


def rot90(img,k):
	"""
	Rotates an image k*90 degrees
	:param img: input image
	:param k: number of times the image is rotated by 90 degrees

	:returns: rotated image
	"""
	return np.rot90(img,k=k)

def flip_ud(img):
	"""
	Flips an image upside-down
	:param img: input image

	:returns: flipped image
	"""
	return np.flipud(img)

def flip_lr(img):
	"""
	Flips an image left-to-right
	:param img: input image

	:returns: flipped image
	"""
	return np.fliplr(img)

def change_brightness(img,bits=8):
	"""
	Randomly change img brightness
	:param img: input image
	:param bits: number of bits to represent a single color value

	:returns: transformed image
	"""
	MAX = get_max(bits)

	brightness = np.random.choice([-MAX//2,-MAX//4,0,MAX//4,MAX//2],1)

	if brightness > 0:
		shadow = brightness
		highlight = MAX
	else:
		shadow = 0
		highlight = MAX + brightness
	alpha_b = (highlight - shadow)/MAX
	gamma_b = shadow

	buf = cv2.addWeighted(img, alpha_b, img, 0, gamma_b)

	return buf

def change_contrast(img,bits=8):
	"""
	Randomly change img contrast
	:param img: input image
	:param bits: number of bits to represent a single color value

	:returns: transformed image
	"""
	MAX = get_max(bits)

	contrast = np.random.choice([-MAX//4,-MAX//8,0,MAX//8,MAX//4],1)


	f = (MAX//2)*(contrast + (MAX//2))/((MAX//2)*(MAX//2-contrast) +1)
	alpha_c = f
	gamma_c = (MAX//2)*(1-f)

	buf = cv2.addWeighted(img, alpha_c, img, 0, gamma_c)

	return buf

def add_gauss_noise(img,bits):
	"""
	Add random gaussian noise to image
	:param img: input image
	:param bits: number of bits to represent a single color value

	:returns: image with noise
	"""
	MAX = get_max(bits)
	noise = img.copy()
	cv2.randn(noise, 0, MAX//2)
	return img + noise

def gamma_correction(img,bits):
	"""
	Image gamma correction with random gamma
	:param img: input image
	:param bits: number of bits to represent a single color value

	:returns: corrected image
	"""
	MAX = get_max(bits)
	gamma = np.random.randint(1,20)/10.
	invGamma = 1.0 / gamma
	table = np.array([((i / float(MAX)) ** invGamma) * MAX
		for i in np.arange(0, (MAX+1))])
 
	return cv2.LUT(img, table)

def apply(aug,img,bits):
	"""
	Maps augmentation name to action
	:param aug: augmentation name
	:param img: input image

	:returns: augmented image
	:raises AugmentationNotFound: exception to indicate that specified augmentation is not recongized
	"""
	if aug == "NO":
		return img
	elif aug == "ROT90":
		return rot90(img,1)
	elif aug == "ROT180":
		return rot90(img,2)
	elif aug == "ROT270":
		return rot90(img,3)
	elif aug == "FLIP_UD":
		return flip_ud(img)
	elif aug == "FLIP_LR":
		return flip_lr(img)
	elif aug == "BRIGHT":
		return change_brightness(img,bits)
	elif aug == "CONTRAST":
		return change_contrast(img,bits)
	elif aug == "GAUSSIAN":
		return add_gauss_noise(img,bits)
	elif aug == "GAMMA":
		return gamma_correction(img,bits)

def reverse(aug,img):
	"""
	Maps augmentation name to reverse action
	:param aug: augmentation name
	:param img: input image

	:returns: reverse of augmented image
	"""
	if aug == "NO":
		return img
	elif aug == "ROT90":
		return rot90(img,-1)
	elif aug == "ROT180":
		return rot90(img,-2)
	elif aug == "ROT270":
		return rot90(img,-3)
	elif aug == "FLIP_UD":
		return flip_ud(img)
	elif aug == "FLIP_LR":
		return flip_lr(img)
	elif aug == "BRIGHT":
		return img
	elif aug == "CONTRAST":
		return img
	elif aug == "GAUSSIAN":
		return img
	elif aug == "GAMMA":
		return img
