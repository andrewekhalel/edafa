
class BasePredictor:
	def __init__(self,model,patch_h,patch_w,augs,*args, **kwargs):
		self.model = model
		self.patch_h = patch_h
		self.patch_w = patch_w
		self.args = args
		self.kwargs = kwargs
		self.augs = augs

	def predict_patch(self,patch):
		raise NotImplementedError('predict_patch() must be implemented')

	def predict_dir(self,path,overlap=0):
		pass