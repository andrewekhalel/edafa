import cv2
import math
import numpy as np


EXTENSIONS = ['jpg','png','tif','tiff']
AUGS = ['NO','ROT90', 'ROT180','ROT270', 'FLIP_UD','FLIP_LR']
MEANS = ['ARITH', 'GEO']

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

def flip_up(img):
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

def apply(aug,img):
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
		return flip_up(img)
	elif aug == "FLIP_LR":
		return flip_lr(img)

def reverse(aug,img):
	"""
	Maps augmentation name to reverse action
	:param aug: augmentation name
	:param img: input image

	:returns: reverse of augmented image
	:raises AugmentationNotFound: exception to indicate that specified augmentation is not recongized
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
		return flip_up(img)
	elif aug == "FLIP_LR":
		return flip_lr(img)
