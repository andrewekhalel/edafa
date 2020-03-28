class UnsupportedDataType(Exception):
	"""
	An exception to indicate data type is unsupported
	"""
	def __init__(self, message):
		self.message = message

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

class ConfigurationUnrecognized(Exception):
	"""
	An exception to indicate passed configuration is unrecognized
	"""
	def __init__(self, message):
		self.message = message
