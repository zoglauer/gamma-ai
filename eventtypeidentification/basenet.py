import tensorflow as tf
import numpy as np

class NPZSaver(object):
	
	def __init__(self, net):
		self._net = net
	
	def save(self, session, f):
		np.savez_compressed(f, **dict((v.name, session.run(v)) for v in self._net.variables))
	
	def restore(self, session, f):
		kwds = np.load(f)
		for v in self._net.variables:
			if v.name in kwds:
				session.run(v.assign(kwds[v.name]))

class BaseNet(object):
	
	def append(self, name, x):
		setattr(x, 'layer_name', name)
		self._layers.append(x)
		return self

	def layer(func):
		def w(self, name, x='', *args, **kwargs):
			with tf.variable_scope(self._name):
				if isinstance(x, list) or isinstance(x, tuple):
					x = [self[i] for i in x]
				elif isinstance(x, str):	
					x = self[x]
				else:
					x, args = self[''], (x,)+args
				x = func(self, name, x, *args, **kwargs)
				for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self._name+'/'+name):
					setattr(x, v.name.split('/')[-1].split(':')[0], v)
				self.append(name, x)
			return self
		return w
	layer = staticmethod(layer)
	
	def __getitem__(self, i):
		if isinstance(i, int) and i < len(self._layers):
			return self._layers[i]
		for l in self._layers:
			if hasattr(l,'layer_name') and l.layer_name == i: 
				return l
		return self.output

	def __init__(self, name, x):
		self._layers = []
		self._name = name
		self.append('input', x)

	def __str__(self): return '\n'.join(
		l.layer_name+'  '+str(l.shape.as_list())+''.join(
			'\n    '+v.name+'  '+str(v.shape.as_list())
			for v in self.variables if l.layer_name in v.name.split('/'))
		for l in self._layers)
	__repr__ = __str__
	def __len__(self): return len(self._layers)
	def __iter__(self): return iter(self._layers)
	@property
	def kernels(self): return [l.kernel for l in self._layers if hasattr(l, 'kernel')]
	@property
	def biases(self): return [l.biases for l in self._layers if hasattr(l, 'biases')]
	@property
	def variables(self): return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self._name)
	@property
	def total_params(self): return sum(reduce(lambda a,b:a*b, v.shape, 1) for v in self.variables)
	@property
	def input(self): return self._layers[0] if self._layers else None # self[0]
	@property
	def output(self): return self._layers[-1] if self._layers else None # self[-1]
	@property
	def saver(self): return tf.train.Saver(self.variables)
	@property
	def npz_saver(self): return NPZSaver(self)
