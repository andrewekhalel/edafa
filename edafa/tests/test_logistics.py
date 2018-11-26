from unittest import TestCase
from edafa import ClassPredictor
import os

class Child(ClassPredictor):
	"""
	Child class to be used in testing
	"""
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

	def predict_patches(self,patches):
		return patches

class LogisticsTester(TestCase):
	def __init__(self, *args, **kwargs):
		super(LogisticsTester, self).__init__(*args, **kwargs)
		self.path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

	def test_pass_json_file(self):
		"""
		Test json configuration file loading
		"""
		p = Child(os.path.join(self.path,"conf/pascal_voc.json"))
		self.assertTrue(p.augs == ["NO",
									"FLIP_UD",
									"FLIP_LR"])
		self.assertTrue(p.mean == "ARITH")
		self.assertTrue(p.bits == 8)

	def test_pass_yaml_file(self):
		"""
		Test yaml configuration file loading
		"""
		p = Child(os.path.join(self.path,"conf/pascal_voc.yaml"))
		self.assertTrue(p.augs == ["NO",
									"FLIP_UD",
									"FLIP_LR"])
		self.assertTrue(p.mean == "ARITH")
		self.assertTrue(p.bits == 8)


	def test_pass_json_string(self):
		"""
		Test configuration as string
		"""
		conf = '{"augs":["NO",\
				"FLIP_UD",\
				"FLIP_LR"],\
				"mean":"ARITH",\
				"bits":8}'
		p = Child(conf)
		self.assertTrue(p.augs == ["NO",
									"FLIP_UD",
									"FLIP_LR"])
		self.assertTrue(p.mean == "ARITH")
		self.assertTrue(p.bits == 8)

	def test_pass_no_augs(self):
		"""
		Test configuration without augs
		"""
		conf = '{"mean":"ARITH",\
				"bits":8}'

		with self.assertWarns(SyntaxWarning):
			p = Child(conf)
			
		self.assertTrue(p.augs == ["NO"])
		self.assertTrue(p.mean == "ARITH")
		self.assertTrue(p.bits == 8)

	def test_pass_no_mean(self):
		"""
		Test configuration without mean
		"""
		conf = '{"augs":["NO"],\
				"bits":8}'
	
		with self.assertWarns(SyntaxWarning):
			p = Child(conf)
			
		self.assertTrue(p.augs == ["NO"])
		self.assertTrue(p.mean == "ARITH")
		self.assertTrue(p.bits == 8)

	def test_pass_no_bits(self):
		"""
		Test configuration without bits
		"""
		conf = '{"augs":["NO"],\
				"mean":"ARITH"}'
			
		with self.assertWarns(SyntaxWarning):
			p = Child(conf)

		self.assertTrue(p.augs == ["NO"])
		self.assertTrue(p.mean == "ARITH")
		self.assertTrue(p.bits == 8)

	def test_pass_invalid_augs(self):
		"""
		Test configuration invalid augs
		"""
		conf = '{"augs":["INVALID"],\
				"mean":"ARITH",\
				"bits":8}'
		
		# TODO: replace Exception with AugmentationUnrecognized
		with self.assertRaises(Exception):
			p = Child(conf)

	def test_pass_invalid_mean(self):
		"""
		Test configuration invalid mean
		"""
		conf = '{"augs":["NO"],\
				"mean":"INVALID",\
				"bits":8}'

		# TODO: replace Exception with MeanUnrecognized
		with self.assertRaises(Exception):
			p = Child(conf)


if __name__ == '__main__':
    unittest.main()