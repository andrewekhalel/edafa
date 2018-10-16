import cv2
import math

def cint(num):
	return int(math.ceil(num))

def add_reflections(img,in_patch_size,out_patch_size):
	O_H = imgs.shape[0]
	O_W = imgs.shape[1]
	
	padding = rint((out_patch_size - in_patch_size)/2.)

	top = padding
	left = padding
	right = rint(in_patch_size - (O_W % in_patch_size)) + padding
	bottom = rint(in_patch_size - (O_H % in_patch_size)) + padding

	return cv2.copyMakeBorder(img,
							top, 
							bottom,
							left, 
							right,
							cv2.BORDER_REFLECT_101)

def apply_aug(img,augs):
	pass