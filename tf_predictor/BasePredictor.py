import tifffile as tif
import os
import cv2
from .utils import *
import numpy as np

class BasePredictor:
	def __init__(self,in_patch_size,out_patch_size,augs,*args, **kwargs):		
		self.in_patch_size = in_patch_size
		self.out_patch_size = out_patch_size
		self.augs = augs
		self.args = args
		self.kwargs = kwargs
		

	def predict_patches(self,patches):
		raise NotImplementedError('predict_patches() must be implemented')

	def normalize(self,img):
		raise NotImplementedError('normalize() must be implemented')

	def postprocess(self,pred):
		raise NotImplementedError('postprocess() must be implemented')

	def predict_dir(self,in_path,out_path,overlap=0):
		for f in os.listdir(in_path):
			if f.endswith('.tif') or f.endswith('.tiff'):
				img = tif.imread(os.path.join(in_path,f))
			else:
				img = cv2.imread(os.path.join(in_path,f))

			O_H = img.shape[0]
			O_W = img.shape[1]

			output = np.zeros(img.shape)
			times = np.zeros((O_H,O_W))

			img = add_reflections(self.normalize(img))

			padding = rint((self.out_patch_size - self.in_patch_size)/2.)

			delta = self.in_patch_size-overlap
			for h in range(0,img.shape[0]-self.out_patch_size +1,delta):
				for w in range(0,img.shape[1]-self.out_patch_size +1,delta):
					in_patch = img[h:(h+self.out_patch_size),
									w:(w+self.out_patch_size),
									:]
					aug_patches = apply_aug(in_patch,self.augs)
					pred = self.predict_patches(aug_patches)

					output[padding:(padding+self.in_patch_size),
							padding:(padding+self.in_patch_size),
							:] = np.mean(pred,axis=0)
					times[padding:(padding+self.in_patch_size),
							padding:(padding+self.in_patch_size)] += 1

			out_filename = f.split('.')[0] + '.png'
			processed = output/times
			cv2.imwrite(os.path.join(out_path,out_filename),processed)
