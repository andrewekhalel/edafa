import tifffile as tif
import os
import cv2
from .utils import *
from .exceptions import *
import numpy as np
from abc import ABC, abstractmethod
import json
import warnings

class BasePredictor(ABC):
	"""
	An abstract class (wrapper for your model) to apply test time augmentation (TTA)
	"""
	def __init__(self,conf_path):
		""" 
		Class constructor

		
		:param conf_path: configuration file path
		"""

		self._parse_conf(conf_path)


		
	@abstractmethod
	def predict_patches(self,patches):
		"""
		Virtual method uses your model to predict patches
		:param patches: input patches to model for prediction

		:return: prediction on these patches
		"""
		pass

	# @abstractmethod
	# def preprocess(self,img):
	# 	"""
	# 	Virtual method to preprocess image before passing to model (normalize, contrast enhancement, ...)
	# 	:param img: input image just after reading it

	# 	:returns: processed image
	# 	"""
	# 	pass

	# @abstractmethod
	# def postprocess(self,pred):
	# 	"""
	# 	Virtual method to postprocess image after model prediction (reverse normalization, clipping, ...)
	# 	:param pred: image predicted using model

	# 	:returns: processed image
	# 	"""
	# 	pass

	def _parse_conf(self,conf_path):
		with open(conf_path) as f:
			loaded = json.load(f)
			if "augs" in  loaded:
				self.augs = loaded["augs"]
				for aug in self.augs:
					if aug not in AUGS:
						raise AugmentationUnrecognized('Unrecognized augmentation: (%s) in configuration file.'%aug)
			else:
				warnings.warn('No "augs" found in configuration file. No augmentations will be used.')
				self.augs = ["NO"]


			if "mean" in loaded:
				self.mean = loaded["mean"]
				if self.mean not in MEANS:
					raise MeanUnrecognized('Unrecognized mean: (%s) in configuration file.'%self.mean)
			else:
				warnings.warn('No "mean" found in configuration file. "ARITH" mean will be used.')
				self.mean = "ARITH"

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

	@abstractmethod
	def reverse_aug(self,aug_patch):
		"""
		Reverse augmentations applied and calculate their combined mean
		:param aug_patch: set of prediction of the model to different augmentations
		
		:returns: single combined patch 
		"""
		pass
			
	@abstractmethod
	def _predict_single(self,img,overlap=0):
		pass

	@abstractmethod
	def predict_images(self,imgs,overlap=0):
		pass
	
	# def predict_dir(self,in_path,out_path,overlap=0,extension='.png'):
	# 	"""
	# 	Predict all images in directory

	# 	:param in_path: directory where original images exist
	# 	:param out_path: directory where predictions should be saved
	# 	:param overlap: overlap size between patches in prediction (default = 0)
	# 	:param extension: extension of saved images (default = '.png')

	# 	"""
	# 	for f in os.listdir(in_path):
	# 		if f.split('.')[-1].lower() not in EXTENSIONS:
	# 			continue

	# 		if f.endswith('.tif') or f.endswith('.tiff'):
	# 			img = tif.imread(os.path.join(in_path,f))
	# 		else:
	# 			img = cv2.imread(os.path.join(in_path,f))

	# 		preprocessed = self.preprocess(img)
	# 		if len(preprocessed.shape) == 4:
	# 			preprocessed = preprocessed[0,:,:,:]

	# 		pred = self.predict_single(preprocessed,overlap=overlap)

	# 		out_filename = f.split('.')[0] + extension
	# 		processed = self.postprocess(pred)
	# 		cv2.imwrite(os.path.join(out_path,out_filename),processed)
