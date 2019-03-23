import numpy as np
import os
import cStringIO
import tensorflow as tf
import random
import zipfile

class ShapeNet40Vox30(object):

	def __init__(self, batch_size=None):

		class Voxels(object):

			def __init__(self, zf, fi, label, category):
				self._zf = zf
				self._fi = fi
				self._label = label
				self._category = category

			@property
			def voxels(self):
				fi = cStringIO.StringIO(self._zf.read(self._fi))
				return np.load(fi)

			@property 
			def label(self):
				return self._label

			@property
			def category(self):
				return self._category

			@property
			def filename(self):
				return self._fi.filename.split('/')[-1]

			def save(self, f=None):
				self.filename if f is None else f
				np.save(f, self.voxels)

		print('Setting up ShapeNet40Vox30 database...')

		self._batch_size = batch_size
		self._mode = 'train'
		self._iters = {}
		
		def get_random_iter(mode):
			while 1:
				order = np.arange(len(self._data[mode]))
				np.random.shuffle(order)
				for i in order:
					yield i

		if not os.path.isfile('volumetric_data_.zip'):
			with zipfile.ZipFile('volumetric_data.zip') as zf:
				zf.extract('volumetric_data_.zip')

		self._zf = zipfile.ZipFile('volumetric_data_.zip')
		
		self._data = {'train': [], 'test': []}
		for i in self._zf.infolist():
			l = i.filename.split('/')
			category = l[0]
			train_or_test = l[2]
			self._data[train_or_test].append((category, i))

		categories = sorted(list(set(c for c,i in self._data['test'])))
		categories = dict(zip(categories, range(len(categories))))
		
		for k in self._data:
			self._data[k] = [Voxels(self._zf, i, categories[c], c) for c,i in self._data[k]]
			self._iters[k] = iter(get_random_iter(k))
			
		self.categories = categories.keys() 
		print('ShapeNet40Vox30 database setup complete!')

	@property
	def num_categories(self):
		return len(self.categories)

	@property
	def train(self):
		self._mode = 'train'
		return self

	@property
	def test(self):
		self._mode = 'test'
		return self

	@property
	def data(self):
		return self._data[self._mode]

	def __len__(self):
		return len(self._data[self._mode])

	def get_batch(self, batch_size=None):
		rn = random.randint
		bs = batch_size if batch_size is not None else self._batch_size
		bs = bs if bs is not None else 25
		voxs = np.zeros([bs, 100,100,50, 1], dtype=np.float32)
		one_hots = np.zeros([bs, self.num_categories], dtype=np.float32)
		data = self._data[self._mode]
		next_int = self._iters[self._mode].next
		for bi in xrange(bs):
			v = data[next_int()]
			d = v.voxels.reshape([98,98,48, 1])
			for axis in 0,1,2: 
				if rn(0,1): 
					d = np.flip(d, axis)
			ox, oy, oz = rn(0,2), rn(0,2), rn(0,2)
			voxs[bi, ox:98+ox,oy:98+oy,oz:48+oz] = d
			one_hots[bi][v.label] = 1
		return voxs, one_hots