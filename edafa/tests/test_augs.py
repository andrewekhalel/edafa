from unittest import TestCase
from edafa import SegPredictor
import os
import cv2
import numpy as np

import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import rot90,flip_ud,flip_lr

class Child(SegPredictor):
	"""
	Child class to be used in testing
	"""
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

	def predict_patches(self,patches):
		return patches

class AugTester(TestCase):
	def __init__(self, *args, **kwargs):
		super(AugTester, self).__init__(*args, **kwargs)
		self.res_dir = os.path.join(os.path.dirname(__file__),'res')
	
	def _read_img(self,fname):
		return cv2.imread(os.path.join(self.res_dir,fname))

	def test_no_aug(self):
		"""
		Test "NO" augmentation
		"""
		img = self._read_img('lena512color.tiff')
		conf ='{"augs":["NO"],\
				"mean":"ARITH",\
				"bits":8\
				}'
		p = Child(in_patch_size=512,out_channels=3,conf=conf)
		self.assertTrue((p.predict_images([img])[0] - img).sum()==0)

	def test_rot90(self):
		"""
		Test "ROT90" augmentation
		"""
		img = self._read_img('lena512color.tiff')
		conf ='{"augs":["ROT90"],\
				"mean":"ARITH",\
				"bits":8\
				}'
		p = Child(in_patch_size=512,out_channels=3,conf=conf)
		self.assertTrue((p.predict_images([img])[0] - img).sum()==0)

	def test_rot180(self):
		"""
		Test "ROT180" augmentation
		"""
		img = self._read_img('lena512color.tiff')
		conf ='{"augs":["ROT180"],\
				"mean":"ARITH",\
				"bits":8\
				}'
		p = Child(in_patch_size=512,out_channels=3,conf=conf)
		self.assertTrue((p.predict_images([img])[0] - img).sum()==0)

	def test_flip_ud(self):
		"""
		Test "FLIP_UD" augmentation
		"""
		img = self._read_img('lena512color.tiff')
		conf ='{"augs":["FLIP_UD"],\
				"mean":"ARITH",\
				"bits":8\
				}'
		p = Child(in_patch_size=512,out_channels=3,conf=conf)
		self.assertTrue((p.predict_images([img])[0] - img).sum()==0)
	
	def test_flip_lr(self):
		"""
		Test "FLIP_LR" augmentation
		"""
		img = self._read_img('lena512color.tiff')
		conf ='{"augs":["FLIP_LR"],\
				"mean":"ARITH",\
				"bits":8\
				}'
		p = Child(in_patch_size=512,out_channels=3,conf=conf)
		self.assertTrue((p.predict_images([img])[0] - img).sum()==0)

	def test_rot90_fn(self):
		"""
		Test rotation function
		"""
		img = self._read_img('lena512color.tiff')
		img_rot = self._read_img('lena512color_rot90.tiff')
		
		self.assertTrue((rot90(img,1)-img_rot).sum() ==0)

	def test_flip_ud_fn(self):
		"""
		Test upside-down flipping function
		"""
		img = self._read_img('lena512color.tiff')
		img_flip_ud = self._read_img('lena512color_flip_ud.tiff')
		
		self.assertTrue((flip_ud(img)-img_flip_ud).sum() ==0)

	def test_flip_lr_fn(self):
		"""
		Test keft-right flipping function
		"""
		img = self._read_img('lena512color.tiff')
		img_flip_lr = self._read_img('lena512color_flip_lr.tiff')
		
		self.assertTrue((flip_lr(img)-img_flip_lr).sum() ==0)

	def _test_bright(self):
		"""
		Test "BRIGHT" augmentation
		"""
		img = self._read_img('lena512color.tiff')
		conf ='{"augs":["BRIGHT"],\
				"mean":"ARITH",\
				"bits":8\
				}'
		p = Child(in_patch_size=512,out_channels=3,conf=conf)
		f, axarr = plt.subplots(2)
		axarr[0].imshow(img[...,::-1])
		axarr[1].imshow(p.predict_images([img])[0][...,::-1].astype(np.uint8))
		plt.show()
				
	def _test_contrast(self):
		"""
		Test "CONTRAST" augmentation
		"""
		img = self._read_img('lena512color.tiff')
		conf ='{"augs":["CONTRAST"],\
				"mean":"ARITH",\
				"bits":8\
				}'
		p = Child(in_patch_size=512,out_channels=3,conf=conf)
		f, axarr = plt.subplots(2)
		axarr[0].imshow(img[...,::-1])
		axarr[1].imshow(p.predict_images([img])[0][...,::-1].astype(np.uint8))
		plt.show()

	def _test_gauss_noise(self):
		"""
		Test "GAUSSIAN" augmentation
		"""
		img = self._read_img('lena512color.tiff')
		conf ='{"augs":["GAUSSIAN"],\
				"mean":"ARITH",\
				"bits":8\
				}'
		p = Child(in_patch_size=512,out_channels=3,conf=conf)
		f, axarr = plt.subplots(2)
		axarr[0].imshow(img[...,::-1])
		axarr[1].imshow(p.predict_images([img])[0][...,::-1].astype(np.uint8))
		plt.show()

	def _test_gamma_correction(self):
		"""
		Test "GAMMA" augmentation
		"""
		img = self._read_img('lena512color.tiff')
		conf ='{"augs":["GAMMA"],\
				"mean":"ARITH",\
				"bits":8\
				}'
		p = Child(in_patch_size=512,out_channels=3,conf=conf)
		f, axarr = plt.subplots(2)
		axarr[0].imshow(img[...,::-1])
		axarr[1].imshow(p.predict_images([img])[0][...,::-1].astype(np.uint8))
		plt.show()