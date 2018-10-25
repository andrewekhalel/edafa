from .BasePredictor import BasePredictor
import numpy as np

class ClassPredictor(BasePredictor):
	def __init__(self,conf_path):
		super().__init__(conf_path=conf_path)

	def reverse_aug(self,aug_patch):
		"""
		Reverse augmentations applied and calculate their combined mean
		:param aug_patch: set of prediction of the model to different augmentations
		
		:returns: single combined patch 
		"""
		if self.mean == "ARITH":
			return np.mean(aug_patch,axis=0)
		elif self.mean == "GEO":
			product = np.prod(aug_patch,axis=0)
			return processed ** (1./len(self.augs))

	def predict_images(self,imgs):
		preds = []
		for img in imgs:
			if len(img.shape) == 4 and img.shape[0] == 1:
				img = img[0,:,:,:]
			pred = self._predict_single(img)
			preds.append(pred)
		return np.array(preds)

	def _predict_single(self,img):
		aug_patches = self.apply_aug(img)
		pred = self.predict_patches(aug_patches)
		return self.reverse_aug(pred)