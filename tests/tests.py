from unittest import TestCase
from edafa import BasePredictor
import os
from abc import abstractmethod

class Child(BasePredictor):
	"""
	Child class to be used in testing
	"""
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)
 
	def reverse_aug(self,aug_patch):
		pass

	def predict_images(self,imgs):
		pass

	def _predict_single(self,patches):
		pass

class Tester(TestCase):
	def __init__(self, *args, **kwargs):
		super(Tester, self).__init__(*args, **kwargs)
		self.path = os.path.join(os.path.dirname(os.path.dirname(__file__)))

	def test_pass_json_loading(self):
		"""
		Test configuration loading
		"""
		p = Child(os.path.join(self.path,"conf/pascal_voc.json"))
		self.assertTrue(p.augs == ["NO",
									"FLIP_UD",
									"FLIP_LR"])
		self.assertTrue(p.mean == "ARITH")

if __name__ == '__main__':
    unittest.main()