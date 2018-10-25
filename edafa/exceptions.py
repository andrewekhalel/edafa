
class AugmentationUnrecognized(Exception):
	"""
	An exception to indicate augmentation is unrecognized
	"""
	def __init__(self, message):
		self.message = message

class MeanUnrecognized(Exception):
	"""
	An exception to indicate mean is unrecognized
	"""
	def __init__(self, message):
		self.message = message
