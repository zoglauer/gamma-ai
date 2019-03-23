import math, datetime
from voxnet import *
from volumetric_data import ShapeNet40Vox30

dataset = ShapeNet40Vox30()
voxnet = VoxNet()

p = dict() # placeholders

p['labels'] = tf.placeholder(tf.float32, [200, 6697])
p['correct_prediction'] = tf.equal(tf.argmax(voxnet[-1], 1), tf.argmax(p['labels'], 1))
p['accuracy'] = tf.reduce_mean(tf.cast(p['correct_prediction'], tf.float32))

num_batches = 2147483647
batch_size = 64

checkpoint_num = int(max([map(float, l.split()) 
	for l in open('checkpoints/accuracies.txt').readlines()], 
	key=lambda x:x[2])[0])

with tf.Session() as session:
	session.run(tf.global_variables_initializer())
	voxnet.npz_saver.restore(session, 'checkpoints/c-{}.npz'.format(checkpoint_num))

	total_accuracy = 0
	for batch_index in xrange(num_batches):

		voxs, labels = dataset.test.get_batch(batch_size)
		feed_dict = {voxnet[0]: voxs, p['labels']: labels}
		total_accuracy += session.run(p['accuracy'], feed_dict=feed_dict)
		test_accuracy = total_accuracy / (batch_index+1)
		if batch_index % 32 == 0:
			print('average test accuracy: {}'.format(test_accuracy))