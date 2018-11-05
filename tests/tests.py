from unittest import TestCase
from edafa import ClassPredictor
import os
from abc import abstractmethod

class Child(ClassPredictor):
	"""
	Child class to be used in testing
	"""
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

	def predict_patches(self,patches):
		pass

class Tester(TestCase):
	def __init__(self, *args, **kwargs):
		super(Tester, self).__init__(*args, **kwargs)
		self.path = os.path.join(os.path.dirname(os.path.dirname(__file__)))

	def test_pass_json_file(self):
		"""
		Test configuration file loading
		"""
		p = Child(os.path.join(self.path,"conf/pascal_voc.json"))
		self.assertTrue(p.augs == ["NO",
									"FLIP_UD",
									"FLIP_LR"])
		self.assertTrue(p.mean == "ARITH")

	def test_pass_json_string(self):
		"""
		Test configuration as string
		"""
		conf = '{"augs":["NO",\
				"FLIP_UD",\
				"FLIP_LR"],\
				"mean":"ARITH"}'
		p = Child(conf)
		self.assertTrue(p.augs == ["NO",
									"FLIP_UD",
									"FLIP_LR"])
		self.assertTrue(p.mean == "ARITH")

if __name__ == '__main__':
    unittest.main()