import tifffile as tif
import os
import cv2
from .utils import *
import numpy as np
import json 
from abc import ABC, abstractmethod

class BasePredictor(ABC):
	"""
	An abstract class (wrapper for your model) to apply test time augmentation (TTA)
	"""
	def __init__(self,in_patch_size,channels,conf_path,out_patch_size=None):
		""" 
		Class constructor

		:param in_patch_size: input patch size (assumes width = height)
		:param channels: number of channels in your model's output (for example number of classes in segmentation)
		:param conf_path: configuration file path
		:param out_patch_size: output patch size in case of no padding (default = in_patch_size)
		"""
		self.in_patch_size = in_patch_size	
		self.channels = channels
		
		with open(conf_path) as f:
			loaded = json.load(f)
			if "augs" in  loaded:
				self.augs = loaded["augs"]
			else:
				self.augs = ["NO"]

			if "mean" in loaded:
				self.mean = loaded["mean"]
			else:
				self.mean = "ARITHMETIC"

		if out_patch_size is None:
			self.out_patch_size = self.in_patch_size
		else:
			self.out_patch_size = out_patch_size
		
	@abstractmethod
	def predict_patches(self,patches):
		"""
		Virtual method uses your model to predict patches
		:param patches: input patches to model for prediction

		:return: prediction on these patches
		"""
		pass

	@abstractmethod
	def preprocess(self,img):
		"""
		Virtual method to preprocess image before passing to model (normalize, contrast enhancement, ...)
		:param img: input image just after reading it

		:returns: processed image
		"""
		pass

	@abstractmethod
	def postprocess(self,pred):
		"""
		Virtual method to postprocess image after model prediction (reverse normalization, clipping, ...)
		:param pred: image predicted using model

		:returns: processed image
		"""
		pass

	def apply_aug(self,img):
		"""
		Apply augmentations to the supplied image
		:param img: original image before augmentation
		
		:returns: a set of augmentations of original image
		"""
		aug_patch = np.zeros((len(self.augs),*img.shape))
		for i,aug in enumerate(self.augs):
			aug_patch[i] = apply(aug,img)
		return aug_patch

	def reverse_aug(self,aug_patch):
		"""
		Reverse augmentations applied and calculate their combined mean
		:param aug_patch: set of prediction of the model to different augmentations
		
		:returns: single combined patch 
		:raises MeanTypeNotFound: exception to indicate that specified mean is not recongized
		"""
		if self.mean == "ARITHMETIC":
			mixed = np.zeros(aug_patch.shape[1:])
			for i,aug in enumerate(self.augs):
				mixed += reverse(aug,aug_patch[i])
			return mixed / len(self.augs)
		elif self.mean == "GEOMETRIC":
			mixed = np.ones(aug_patch.shape[1:])
			for i,aug in enumerate(self.augs):
				mixed *= reverse(aug,aug_patch[i])
			return mixed ** (1./len(self.augs))
		else:
			raise MeanTypeNotFound("%s is not a valid mean type! Currently supported types: GEOMETRIC, ARITHMETIC"%self.mean)

	def predict_dir(self,in_path,out_path,overlap=0,extension='.png'):
		"""
		Predict all images in directory

		:param in_path: directory where original images exist
		:param out_path: directory where predictions should be saved
		:param overlap: overlap size between patches in prediction (default = 0)
		:param extension: extension of saved images (default = '.png')
		"""
		for f in os.listdir(in_path):
			if f.split('.')[-1].lower() not in EXTENSIONS:
				continue

			if f.endswith('.tif') or f.endswith('.tiff'):
				img = tif.imread(os.path.join(in_path,f))
			else:
				img = cv2.imread(os.path.join(in_path,f))

			preprocessed = self.preprocess(img)
			if len(preprocessed.shape) == 4:
				preprocessed = preprocessed[0,:,:,:]


			output = np.zeros((*preprocessed.shape[:2],self.channels))
			times = np.zeros(preprocessed.shape[:2])

			img = add_reflections(preprocessed,self.in_patch_size,self.out_patch_size)

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
					if overlap == 0:
						assert np.sum(times) == times.shape[0]*times.shape[1]
			out_filename = f.split('.')[0] + extension
			processed = self.postprocess(output/times[:,:,np.newaxis])
			cv2.imwrite(os.path.join(out_path,out_filename),processed)
