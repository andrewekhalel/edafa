from unittest import TestCase
from tf_predictor import BasePredictor
import os

class Tester(TestCase):
	def __init__(self, *args, **kwargs):
		super(Tester, self).__init__(*args, **kwargs)
		self.path = os.path.join(os.path.dirname(os.path.dirname(__file__)))

	def test_pass_args(self):
		p = BasePredictor(128,128,os.path.join(self.path,"augs/d4.json"),0,1,2)
		self.assertTrue(p.args[0] == 0)
		self.assertTrue(p.args[1] == 1)
		self.assertTrue(p.args[2] == 2)

	def test_pass_kwargs(self):
		p = BasePredictor(128,128,os.path.join(self.path,"augs/d4.json"),a=0,b=1)
		self.assertTrue(p.kwargs['a'] == 0)
		self.assertTrue(p.kwargs['b'] == 1)

	def test_pass_json_loading(self):
		p = BasePredictor(128,128,os.path.join(self.path,"augs/d4.json"))
		self.assertTrue(p.augs == ["NO",
									"ROT90",
									"ROT180",
									"ROT270",
									"FLIP_UP",
									"FLIP_LR"])

if __name__ == '__main__':
    unittest.main()