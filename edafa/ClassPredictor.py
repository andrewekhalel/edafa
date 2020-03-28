from .BasePredictor import BasePredictor
from .exceptions import UnsupportedDataType
import numpy as np

class ClassPredictor(BasePredictor):
	def __init__(self,conf):
		"""
		Initialize class
		
		:param conf: configuration (json string or file path)
		"""
		super().__init__(conf=conf)

	def reverse_aug(self,aug_patch):
		"""
		Reverse augmentations applied and calculate their combined mean

		:param aug_patch: set of prediction of the model to different augmentations
		
		:returns: single combined patch 
		"""
		if isinstance(aug_patch,np.ndarray):
			if self.mean == "ARITH":
				return np.mean(aug_patch,axis=0)
			elif self.mean == "GEO":
				product = np.prod(aug_patch,axis=0)
				return product ** (1./len(self.augs))
		elif isinstance(aug_patch,list):
			try:
				aug_patch = np.array(aug_patch)
				return np.mean(aug_patch,axis=0)
			except:	
				averaged = []			
				for output in aug_patch:
					averaged.append([sum(e)/len(e) for e in zip(*output)])
				return averaged
				
		else:
			raise UnsupportedDataType('Data type "%s" produced by your model is not supported.\
										list and numpy arrays are the only supported types.'%aug_patch.dtype)


	def _predict_single(self,img):
		"""
		predict single image
		
		:param img: image to predict

		:return: prediction on the image
		"""
		aug_patches = self.apply_aug(img)
		pred = self.predict_patches(aug_patches)
		return self.reverse_aug(pred)