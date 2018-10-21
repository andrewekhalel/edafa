import cv2
import math
import numpy as np

class AugmentationNotFound(Exception):
	def __init__(self, message):
		self.message = message

class MeanTypeNotFound(Exception):
	def __init__(self, message):
		self.message = message

def cint(num):
	return int(math.ceil(num))

def rint(num):
	return int(round(num))

def add_reflections(img,in_patch_size,out_patch_size):
	O_H = img.shape[0]
	O_W = img.shape[1]
	
	padding = rint((out_patch_size - in_patch_size)/2.)

	top = padding
	left = padding
	right = rint(in_patch_size - (O_W % in_patch_size)) + padding
	bottom = rint(in_patch_size - (O_H % in_patch_size)) + padding
	if top ==0 or left ==0 or right ==0 or bottom==0:
		return img
	return cv2.copyMakeBorder(img,
							top, 
							bottom,
							left, 
							right,
							cv2.BORDER_REFLECT_101)


def rot90(img,k):
	return np.rot90(img,k=k)

def flip_up(img):
	return np.flipud(img)

def flip_lr(img):
	return np.fliplr(img)

def apply(aug,img):
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
	else:
		raise AugmentationNotFound("%s is not a valid augmentation!"%aug)

def reverse(aug,img):
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
	else:
		raise AugmentationNotFound("%s is not a valid augmentation!"%aug)
