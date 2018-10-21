from unittest import TestCase
from tta_predictor import BasePredictor
import os

class Child(BasePredictor):
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

	def test_pass_args_kwargs(self):
		p = Child(128,64,1,os.path.join(self.path,"conf/rotates.json"),0,1,a=0,b=1)
		self.assertTrue(p.args[0] == 0)
		self.assertTrue(p.args[1] == 1)
		self.assertTrue(p.in_patch_size == 128)
		self.assertTrue(p.out_patch_size == 64)
		self.assertTrue(p.kwargs['a'] == 0)
		self.assertTrue(p.kwargs['b'] == 1)

	def test_pass_json_loading(self):
		p = Child(128,128,1,os.path.join(self.path,"conf/rotates.json"))
		self.assertTrue(p.augs == ["NO",
									"ROT90",
									"ROT180",
									"ROT270"])

if __name__ == '__main__':
    unittest.main()