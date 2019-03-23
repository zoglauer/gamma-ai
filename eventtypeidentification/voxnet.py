from basenet import *
from functools import reduce

class VoxNet(BaseNet):

	@BaseNet.layer
	def conv3d(self, name, x, num_filters, filter_size, stride, padding='VALID', relu_alpha=0.1):
		with tf.variable_scope(name) as scope:
			x = tf.layers.conv3d(x, num_filters, filter_size, stride, padding)
			x = tf.layers.batch_normalization(x, training=self.training)
			return tf.maximum(x, relu_alpha * x) # leaky relu
	
	@BaseNet.layer
	def max_pool3d(self, name, x, size=2, stride=2, padding='VALID'):
		return tf.layers.max_pooling3d(x, size, stride, padding)

	@BaseNet.layer
	def avg_pool3d(self, name, x, size=2, stride=2, padding='VALID'):
		return tf.layers.avg_pooling3d(x, size, stride, padding)

	@BaseNet.layer
	def fc(self, name, x, num_outputs, batch_norm=True, relu=True):
		with tf.variable_scope(name) as scope:
			x = tf.layers.dense(tf.reshape(x, 
				[-1, reduce(lambda a,b:a*b, x.shape.as_list()[1:])]), num_outputs)
			if batch_norm: x = tf.layers.batch_normalization(x, training=self.training)
			if relu: x = tf.nn.relu(x)
			return x
	
	@BaseNet.layer
	def softmax(self, name, x): return tf.nn.softmax(x)

	def __init__(self, bSize, xSize, ySize, zSize, MaxLabel, voxnet_type='all_conv'):
		self.training = tf.placeholder_with_default(False, shape=None)
		super(VoxNet, self).__init__('voxnet', 
			tf.placeholder(tf.float32, [None, xSize, ySize, zSize, 1]) )


		if voxnet_type == 'original':
			self.conv3d('conv1', 32, 5, 2)
			self.conv3d('conv2', 32, 3, 1)
			self.max_pool3d('max_pool')
			self.fc('fc1', 128)
			print("here")
			self.fc('fc2', MaxLabel, batch_norm=False, relu=False)
			self.softmax('softmax')

		elif voxnet_type == 'all_conv':
			self.conv3d('conv1', 64, 5, 2)
			self.conv3d('conv2', 64, 3, 1)
			self.conv3d('conv3', 128, 2, 2)
			self.conv3d('conv4', 128, 2, 2)
			self.fc('fc1', 128)
			self.fc('fc2', MaxLabel, batch_norm=False, relu=False)
			self.softmax('softmax')
		
if __name__ == '__main__':
	voxnet = VoxNet()
	print(voxnet)
	print('\nTotal number of parameters: {}'.format(voxnet.total_params))
