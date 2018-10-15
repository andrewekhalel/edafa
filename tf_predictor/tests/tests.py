from unittest import TestCase
from tf_predictor import BasePredictor

class Tester(TestCase):
	def __init__(self, *args, **kwargs):
		super(Tester, self).__init__(*args, **kwargs)


	def test_pass_args(self):
		p = BasePredictor(None,128,128,None,0,1,2)
		self.assertTrue(p.args[0] == 0)
		self.assertTrue(p.args[1] == 1)
		self.assertTrue(p.args[2] == 2)

	def test_pass_kwargs(self):
		p = BasePredictor(None,128,128,None,a=0,b=1)
		self.assertTrue(p.kwargs['a'] == 0)
		self.assertTrue(p.kwargs['b'] == 1)

if __name__ == '__main__':
    unittest.main()