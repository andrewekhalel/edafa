import tifffile as tif
import os
import cv2
from .utils import *
import numpy as np
import json 
from abc import ABC, abstractmethod

class BasePredictor(ABC):
	def __init__(self,in_patch_size,out_patch_size,augs_path,*args, **kwargs):		
		self.in_patch_size = in_patch_size
		self.out_patch_size = out_patch_size
		with open(augs_path) as f:
			self.augs = json.load(f)["augs"]
		self.args = args
		self.kwargs = kwargs
		
	@abstractmethod
	def predict_patches(self,patches):
		pass

	@abstractmethod
	def preprocess(self,img):
		pass

	@abstractmethod
	def postprocess(self,pred):
		pass

	def apply_aug(self,img):
		aug_patch = np.zeros((len(self.augs),*img.shape))
		for i,aug in enumerate(self.augs):
			aug_patch[i] = apply(aug,img)
		return aug_patch

	def reverse_aug(self,aug_patch):
		mixed = np.zeros(*aug_patch.shape[1:])
		for i,aug in enumerate(self.augs):
			mixed += reverse(aug,aug_patch[i])
		return mixed / len(self.augs)
		

	def predict_dir(self,in_path,out_path,overlap=0,extension='.png'):
		for f in os.listdir(in_path):
			if f.endswith('.tif') or f.endswith('.tiff'):
				img = tif.imread(os.path.join(in_path,f))
			else:
				img = cv2.imread(os.path.join(in_path,f))

			O_H = img.shape[0]
			O_W = img.shape[1]

			output = np.zeros(img.shape)
			times = np.zeros((O_H,O_W))

			img = add_reflections(self.preprocess(img),self.in_patch_size,self.out_patch_size)

			padding = rint((self.out_patch_size - self.in_patch_size)/2.)

			delta = self.in_patch_size-overlap
			for h in range(0,img.shape[0]-self.out_patch_size +1,delta):
				for w in range(0,img.shape[1]-self.out_patch_size +1,delta):
					in_patch = img[h:(h+self.out_patch_size),
									w:(w+self.out_patch_size),
									:]
					aug_patches = self.apply_aug(in_patch)
					pred = self.predict_patches(aug_patches)
					pred = self.reverse_aug(pred)

					output[padding:(padding+self.in_patch_size),
							padding:(padding+self.in_patch_size),
							:] = pred
					times[padding:(padding+self.in_patch_size),
							padding:(padding+self.in_patch_size)] += 1

			out_filename = f.split('.')[0] + extension
			processed = self.postprocess(output/times)
			cv2.imwrite(os.path.join(out_path,out_filename),processed)
