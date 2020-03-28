from unittest import TestCase
from edafa import ClassPredictor
import numpy as np
import os

class Child(ClassPredictor):
	"""
	Child class to be used in testing
	"""
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

	def predict_patches(self,patches):
		count = patches.shape[0]
		return [[[0]*10 for _ in range(count)],
				[[1]*9 for _ in range(count)],
				[[2]*8 for _ in range(count)]]

class MultiOutputTester(TestCase):
	def __init__(self, *args, **kwargs):
		super(MultiOutputTester, self).__init__(*args, **kwargs)
		self.path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

	def test_multi_output(self):
		"""
		Test Multi-output handling
		"""
		conf = '{"augs":["NO", "FLIP_LR"],\
				"mean":"ARITH",\
				"bits":8}'
		
		
		p = Child(conf)
		output = p.predict_images([np.random.rand(3,2,3)])
		self.assertTrue((output==[[0]*10,[1]*9,[2]*8]).all())