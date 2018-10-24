from unittest import TestCase
from edafa import BasePredictor
import os

class Child(BasePredictor):
	"""
	Child class to be used in testing
	"""
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

	def preprocess(self,img):
		pass

	def postprocess(self,pred):
		pass

	def predict_patches(self,patches):
		pass

class Tester(TestCase):
	def __init__(self, *args, **kwargs):
		super(Tester, self).__init__(*args, **kwargs)
		self.path = os.path.join(os.path.dirname(os.path.dirname(__file__)))

	def test_pass_json_loading(self):
		"""
		Test configuration loading
		"""
		p = Child(128,1,os.path.join(self.path,"conf/pascal_voc.json"))
		self.assertTrue(p.augs == ["NO",
									"FLIP_UD",
									"FLIP_LR"])
		self.assertTrue(p.mean == "ARITHMETIC")

if __name__ == '__main__':
    unittest.main()