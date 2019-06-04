import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd


def plot_testing_acc(x, y):
	df = pd.DataFrame({'x': x, 'y': y, 'z': y})
	f1 = plt.figure(1)
	plt.plot('x','y',data=df,  marker='o', color='blue')
	plt.title("Testing accuracy vs Number of Batches (100s)")
	plt.xlabel("Number of Batches")
	plt.ylabel("Test Accuracy")
	f1.show()

def plot_training_acc(x, y):
	df = pd.DataFrame({'x': x, 'y': y, 'z': y})
	f2= plt.figure(2)
	plt.plot('x','y',data=df,  marker='o', color='green')
	plt.title("Training Accuracy vs Number of Batches (100s)")
	plt.xlabel("Number of Batches")
	plt.ylabel("Train Accuracy")
	f2.show()
	

def plot_training_vs_test(x, y):
	df = pd.DataFrame({'x': x, 'y': y, 'z': y})
	f3= plt.figure(3)
	plt.plot('x','y',data=df,  marker='o', color='mediumvioletred')
	plt.title("Testing Accuracy vs Training Accuracy")
	plt.xlabel("Training Accuracy")
	plt.ylabel("Testing Accuracy")
	f3.show()

#only care about labels 0, 1, 2, 4
#can have a line plot of compton events 
def plot_labels_accuracy_compton(batches, heights):
	label_0_height = heights[0]
	label_1_height = heights[1]
	label_2_height = heights[2]
	label_4_height = heights[4]

	df_0 = pd.DataFrame({'x0': batches, 'y0': label_0_height})
	df_1 = pd.DataFrame({'x1': batches, 'y1': label_1_height})
	df_2 = pd.DataFrame({'x2': batches, 'y2': label_2_height})
	df_4 = pd.DataFrame({'x4': batches, 'y4': label_4_height})
	f4 = plt.figure(4)
	plt.plot('x0','y0',data=df_0,  marker='o', color='red', label = '00')
	plt.plot('x1', 'y1', data = df_1, marker='o', color='blue', label = '01')
	plt.plot('x2', 'y2', data = df_2, marker='o', color='green', label = '02')
	plt.plot('x4', 'y4', data = df_4, marker='o', color='orange', label = '04')
	plt.legend(loc='best')
	f4.show()
	plt.title("Compton Event Accuracy vs Number of Batches (100s)")
	plt.xlabel("Number of Batches")
	plt.ylabel("Compton Event Accuracy")

#care about 10, 11, 12, 14
#pair events 
def plot_labels_accuracy_pair(batches, heights):
	label_10_height = heights[10]
	label_11_height = heights[11]
	label_12_height = heights[12]
	label_14_height = heights[14]

	df_10 = pd.DataFrame({'x10': batches, 'y10': label_10_height})
	df_11 = pd.DataFrame({'x11': batches, 'y11': label_11_height})
	df_12 = pd.DataFrame({'x12': batches, 'y12': label_12_height})
	df_14 = pd.DataFrame({'x14': batches, 'y14': label_14_height})
	f5 = plt.figure(5)
	plt.plot('x10','y10',data=df_10,  marker='o', color='red', label = '10')
	plt.plot('x11', 'y11', data = df_11, marker='o', color='blue', label = '11')
	plt.plot('x12', 'y12', data = df_12, marker='o', color='green', label = '12')
	plt.plot('x14', 'y14', data = df_14, marker='o', color='orange', label = '14')
	plt.legend(loc='best')
	f5.show()
	plt.title("Pair Event Accuracy vs Number of Batches (100s)")
	plt.xlabel("Number of Batches")
	plt.ylabel("Pair Event Accuracy")

	
if __name__=="__main__":
	batches_test_train = []
	training_acc = []
	testing_acc= []
	
	with open('accuracies.txt') as f:
		for line in f:
			split = line.split()
			batches_test_train.append(int(split[0]))
			training_acc.append(float(split[1]))
			testing_acc.append(float(split[2]))

	num_labels = 15
	batches_labels = []
	label_accuracies = []
	for _ in range(num_labels):
		label_accuracies.append([])
	with open('accuracies_labels.txt') as f2:
		for line in f2: 
			split = line.split()
			batches_labels.append(int(split[0]))

			for i in range(0, num_labels):
				label_accuracies[i].append(float(split[i+1]))
	


	plot_training_acc(batches_test_train, training_acc)
	plot_testing_acc(batches_test_train, testing_acc)
	train_test = zip(training_acc, testing_acc)
	sort = sorted(train_test, key = lambda tup: tup[0])
	print(sort)
	unzipped = list(zip(*sort))
	print(unzipped)

	plot_training_vs_test(list(unzipped[0]), list(unzipped[1]))

	plot_labels_accuracy_compton(batches_labels, label_accuracies)
	plot_labels_accuracy_pair(batches_labels, label_accuracies)

	plt.show()
	