class MockObject:

	def __init__(self, obj_type):
		self.obj_type = obj_type

		if obj_type == 'rectangle':
			self.xs = [-0.0025, 0.0025]
			self.ys = [-0.0125, 0.0125]
			self.height = 0.01
		elif obj_type == 'sphere':
			self.xs = [-0.005, 0.005]
			self.ys = [-0.005, 0.005]
			self.height = 0.005
		elif obj_type == 'cylinder':
			self.xs = [-0.005, 0.005]
			self.ys = [-0.005, 0.005]
			self.height = 0.005
		if obj_type == 'ellipsoid':
			self.xs = [-0.0025, 0.0025]
			self.ys = [-0.0100, 0.0100]
			self.height = 0.01

	def get_x(self):
		return self.xs

	def get_y(self):
		return self.ys

	def get_height(self):
		return self.height
