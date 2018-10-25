from .BasePredictor import BasePredictor
import numpy as np
from .utils import *

class SegPredictor(BasePredictor):
	def __init__(self,in_patch_size,out_channels,conf_path,out_patch_size=None):
		"""

		:param in_patch_size: input patch size (assumes width = height)
		:param out_channels: number of channels in your model's output (for example number of classes in segmentation)
		:param conf_path: configuration file path
		:param out_patch_size: output patch size in case of no padding (default = in_patch_size)

		"""
		super().__init__(conf_path=conf_path)

		self.in_patch_size = in_patch_size	
		self.out_channels = out_channels

		if out_patch_size is None:
			self.out_patch_size = self.in_patch_size
		else:
			self.out_patch_size = out_patch_size


	def reverse_aug(self,aug_patch):
		"""
		Reverse augmentations applied and calculate their combined mean
		:param aug_patch: set of prediction of the model to different augmentations
		
		:returns: single combined patch 
		"""
		if self.mean == "ARITH":
			mixed = np.zeros(aug_patch.shape[1:])
			for i,aug in enumerate(self.augs):
				mixed += reverse(aug,aug_patch[i])
			return mixed / len(self.augs)
		elif self.mean == "GEO":
			mixed = np.ones(aug_patch.shape[1:])
			for i,aug in enumerate(self.augs):
				mixed *= reverse(aug,aug_patch[i])
			return mixed ** (1./len(self.augs))


	def predict_images(self,imgs,overlap=0):
		preds = []
		for img in imgs:
			if len(img.shape) == 4 and img.shape[0] == 1:
				img = img[0,:,:,:]
			pred = self._predict_single(img,overlap=overlap)
			preds.append(pred)
		return np.array(preds)

	def _predict_single(self,img,overlap=0):
		output = np.zeros((*img.shape[:2],self.out_channels))
		times = np.zeros(img.shape[:2])

		img = add_reflections(img,self.in_patch_size,self.out_patch_size)

		padding = rint((self.in_patch_size - self.out_patch_size)/2.)

		delta = self.in_patch_size-overlap
		for h in range(0,img.shape[0]-self.in_patch_size +1,delta):
			for w in range(0,img.shape[1]-self.in_patch_size +1,delta):
				in_patch = img[h:(h+self.in_patch_size),
								w:(w+self.in_patch_size),
								:]
				aug_patches = self.apply_aug(in_patch)
				pred = self.predict_patches(aug_patches)
				pred = self.reverse_aug(pred)

				output[padding:(padding+self.out_patch_size),
						padding:(padding+self.out_patch_size),
						:] = pred

				times[padding:(padding+self.out_patch_size),
						padding:(padding+self.out_patch_size)] += 1
		
		return (output/times[:,:,np.newaxis])

