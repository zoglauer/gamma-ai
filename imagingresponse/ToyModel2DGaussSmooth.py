###################################################################################################
#
# ToyModel2DGauss.py
#
# Copyright (C) by Andreas Zoglauer & contributors.
# All rights reserved.
#
# Please see the file LICENSE in the main repository for the copyright-notice.
#
###################################################################################################


# TODO: There is an unknown memeory leak


###################################################################################################

import tensorflow as tf
import numpy as np
import random

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import signal
import sys
import time
import math
import csv

from scipy import signal as sciSignal

print("\nToyModel: (x,y) --> exp(-(x-x0)^2/s0^2)*exp(-(y-y0)^2/s1^2), random) for each x, y in [-1, 1]\n")

gMinXY = -1
gMaxXY = +1
gSigmaX = 0.1
gSigmaY = 0.2

gTrainingGridXY = 30;
gGridCenters = np.zeros(gTrainingGridXY)
for x in range(0, gTrainingGridXY):
    gGridCenters[x] = gMinXY + (x+0.5)*(gMaxXY-gMinXY)/gTrainingGridXY

# Set data parameters
InputDataSpaceSize = 2
OutputDataSpaceSize = gTrainingGridXY*gTrainingGridXY

# SET NUMBER OF NETWORKS @IMPORTANT
numNetworks = 5

SubBatchSize = 1024

NTrainingBatches = 1
TrainingBatchSize = NTrainingBatches*SubBatchSize * numNetworks

NTestingBatches = 1
TestBatchSize = NTestingBatches*SubBatchSize

###################################################################################################
# Step 2: Global functions
###################################################################################################

# First take care of Ctrl-C
Interrupted = False
NInterrupts = 0
def signal_handler(signal, frame):
    print("You pressed Ctrl+C!")
    global Interrupted
    Interrupted = True
    global NInterrupts
    NInterrupts += 1
    if NInterrupts >= 3:
        print("Aborting!")
        sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

###################################################################################################
# Step 3: Create the training, test & verification data sets
###################################################################################################

print("Info: Creating %i data sets" % (TrainingBatchSize + TestBatchSize))

def CreateRandomResponsePoint(PosX, PosY):
    return (PosX + random.gauss(PosX, gSigma), random.uniform(gMinXY, gMaxXY))

def CreateFullResponse(PosX, PosY):
    Out = np.zeros(shape=(1, OutputDataSpaceSize))
    for x in range(0, gTrainingGridXY):
        for y in range(0, gTrainingGridXY):
            Out[0, x + y*gTrainingGridXY] = math.exp(-math.pow(PosX-gGridCenters[x], 2)/math.pow(gSigmaX, 2))*math.exp(-math.pow(PosY-gGridCenters[y], 2)/math.pow(gSigmaY, 2))
            #print("{} = {}".format(Out[0, x + y*gTrainingGridXY], math.exp(-math.pow(PosX-gGridCenters[x], 2)/math.pow(gSigma, 2))))
    return Out

XTrain = np.zeros(shape=(TrainingBatchSize, InputDataSpaceSize))
YTrain = np.zeros(shape=(TrainingBatchSize, OutputDataSpaceSize))
for i in range(0, TrainingBatchSize):
    if i > 0 and i % 128 == 0:
        print("Training set creation: {}/{}".format(i, TrainingBatchSize))
    XTrain[i,0] = random.uniform(gMinXY, gMaxXY)
    XTrain[i,1] = random.uniform(gMinXY, gMaxXY)
    YTrain[i,] = CreateFullResponse(XTrain[i,0], XTrain[i,1])

XTest = np.zeros(shape=(TestBatchSize, InputDataSpaceSize))
YTest = np.zeros(shape=(TestBatchSize, OutputDataSpaceSize))
for i in range(0, TestBatchSize):
    if i > 0 and i % 128 == 0:
        print("Testing set creation: {}/{}".format(i, TestBatchSize))
    XTest[i,0] = random.uniform(gMinXY, gMaxXY)
    XTest[i,1] = random.uniform(gMinXY, gMaxXY)
    YTest[i,] = CreateFullResponse(XTest[i,0], XTest[i,1])

XSingle = XTest[0:1]
YSingle = YTest[0:1]

#fig = plt.figure()
#ax = fig.gca(projection='3d')

#XV = gGridCenters
#YV = gGridCenters
#XV,YV = np.meshgrid(XV,YV)
#Z = YSingle.reshape(gTrainingGridXY, gTrainingGridXY)

#surf = ax.plot_surface(XV, YV, Z)  #, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

#plt.show()
#plt.pause(0.001)

#input("Press [enter] to EXIT")
#sys.exit()

###################################################################################################
# Step 4: Setting up the neural network
###################################################################################################

def CreateNeuralNetwork():
    print("Info: Setting up neural network...")

    # Placeholders
    print("      ... placeholders ...")
    X = tf.placeholder(tf.float32, [None, InputDataSpaceSize], name="X")
    Y = tf.placeholder(tf.float32, [None, OutputDataSpaceSize], name="Y")

    # Layers: 1st hidden layer X1, 2nd hidden layer X2, etc.
    print("      ... hidden layers ...")
    H = tf.contrib.layers.fully_connected(X, 10) #, activation_fn=tf.nn.relu6, weights_initializer=tf.truncated_normal_initializer(0.0, 0.1), biases_initializer=tf.truncated_normal_initializer(0.0, 0.1))
    H = tf.contrib.layers.fully_connected(H, 50) #, activation_fn=tf.nn.relu6, weights_initializer=tf.truncated_normal_initializer(0.0, 0.1), biases_initializer=tf.truncated_normal_initializer(0.0, 0.1))
    H = tf.contrib.layers.fully_connected(H, 100) #, activation_fn=tf.nn.relu6, weights_initializer=tf.truncated_normal_initializer(0.0, 0.1), biases_initializer=tf.truncated_normal_initializer(0.0, 0.1))
    H = tf.contrib.layers.fully_connected(H, 700) #, activation_fn=tf.nn.relu6, weights_initializer=tf.truncated_normal_initializer(0.0, 0.1), biases_initializer=tf.truncated_normal_initializer(0.0, 0.1))
    H = tf.contrib.layers.fully_connected(H, 1000) #, activation_fn=tf.nn.relu6, weights_initializer=tf.truncated_normal_initializer(0.0, 0.1), biases_initializer=tf.truncated_normal_initializer(0.0, 0.1))

    print("      ... output layer ...")
    Output = tf.contrib.layers.fully_connected(H, OutputDataSpaceSize, activation_fn=None)

    # Loss function
    print("      ... loss function ...")
    #LossFunction = tf.reduce_sum(np.abs(Output - Y)/TestBatchSize)
    LossFunction = tf.reduce_sum(tf.pow(Output - Y, 2))/TestBatchSize

    # Minimizer
    print("      ... minimizer ...")
    Trainer = tf.train.AdamOptimizer().minimize(LossFunction)

    # Create and initialize the session
    print("      ... session ...")
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print("      ... writer ...")
    writer = tf.summary.FileWriter("OUT_ToyModel2DGauss", sess.graph)
    writer.close()

    # Add ops to save and restore all the variables.
    print("      ... saver ...")
    Saver = tf.train.Saver()

    return sess, X, Y, Output, Trainer

sess, X, Y, Output, Trainer = CreateNeuralNetwork()
sess2, X2, Y2, Output2, Trainer2 = CreateNeuralNetwork()

sessList = []
XList = []
YList = []
OutputList = []
TrainerList = []

for i in range(numNetworks):
    print("Info: Creating Neural Network Object #" + str(i))
    sessVar, XVar, YVar, OutputVar, TrainerVar = CreateNeuralNetwork()
    sessList.append(sessVar)
    XList.append(XVar)
    YList.append(YVar)
    OutputList.append(OutputVar)
    TrainerList.append(TrainerVar)

###################################################################################################
# Step 3: Training and evaluating the network
###################################################################################################

print("Info: Training and evaluating the network")

# Train the network
Timing = time.process_time()

TimesNoImprovement = 0
BestMeanSquaredError = sys.float_info.max

def CheckPerformance():
    global TimesNoImprovement
    global BestMeanSquaredError

    MeanSquaredError = 0
    total = 0
    for i in range(numNetworks):
        sess = sessList[i]
        X = XList[i]
        Output = OutputList[i]
        total += sess.run(tf.nn.l2_loss(Output - YTest)/TestBatchSize,  feed_dict={X: XTest})

    MeanSquaredError = total / numNetworks

    print("Iteration {} - MSE of test data: {}".format(Iteration, MeanSquaredError))

    if MeanSquaredError <= BestMeanSquaredError:    # We need equal here since later ones are usually better distributed
        BestMeanSquaredError = MeanSquaredError
        TimesNoImprovement = 0

        XSingle = XTest[0:1]
        YSingle = YTest[0:1]

        total = 0
        for i in range(numNetworks):
            sess = sessList[i]
            X = XList[i]
            Output = OutputList[i]
            total += sess.run(Output, feed_dict={X: XSingle})

        YOutSingle = total / numNetworks

        # POWER SPECTRUM OUTPUT
        # """
        fs = 10e3
        N = 1e5
        amp = 2*np.sqrt(2)
        freq = 1234.0
        noise_power = 0.001 * fs / 2
        time = np.arange(N) / fs
        # x = amp*np.sin(2*np.pi*freq*time)
        # x += np.random.normal(scale=np.sqrt(noise_power), size=time.shape)

        """print("Alpha")
        print(len(alpha))

        print("X")
        x = np.array(YOutSingle.tolist())
        x = np.reshape(x, len(x[0]))
        print(len(x))"""

        fig = plt.figure(1)
        plt.clf()
        x = np.array(YSingle.tolist())
        x = np.reshape(x, len(x[0]))
        f, Pxx_den = sciSignal.periodogram(x, fs)
        plt.semilogy(f, Pxx_den)
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')

        fig2 = plt.figure(2)
        plt.clf()
        x = np.array(YOutSingle.tolist())
        x = np.reshape(x, len(x[0]))
        f, Pxx_den = sciSignal.periodogram(x, fs)
        plt.semilogy(f, Pxx_den)
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')

        plt.ion()
        fig.show()
        fig2.show()
        plt.pause(0.001)

        """
        XV,YV = np.meshgrid(gGridCenters, gGridCenters)

        fig = plt.figure(1)
        plt.clf()
        ax = fig.gca(projection='3d')
        ZV = YSingle.reshape(gTrainingGridXY, gTrainingGridXY)
        surf = ax.plot_surface(XV, YV, ZV)  #, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        fig = plt.figure(2)
        plt.clf()
        ax = fig.gca(projection='3d')
        ZV = YOutSingle.reshape(gTrainingGridXY, gTrainingGridXY)
        surf = ax.plot_surface(XV, YV, ZV)  #, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        plt.ion()
        plt.show()
        plt.pause(0.001)
        """

    else:
        TimesNoImprovement += 1

# Main training and evaluation loop
MaxIterations = 50000
for Iteration in range(0, MaxIterations):
    # Take care of Ctrl-C
    if Interrupted == True: break

    # Train
    for Batch in range(0, NTrainingBatches):
        if Interrupted == True: break

        #if Batch%8 == 0:
        #   print("Iteration %6d, Batch %4d)" % (Iteration, Batch))

        Start = Batch * SubBatchSize
        Stop = (Batch + 1) * SubBatchSize

        for i in range(numNetworks):
            newStart = i * Stop
            newStop = (i + 1) * Stop
            sess = sessList[i]
            X = XList[i]
            Y = YList[i]
            Output = OutputList[i]
            Trainer = TrainerList[i]
            sess.run(Trainer, feed_dict={X: XTrain[newStart:newStop], Y: YTrain[newStart:newStop]})

    # Check performance: Mean squared error
    if Iteration > 0 and Iteration % 20 == 0:
        CheckPerformance()

    if TimesNoImprovement == 100:
        print("No improvement for 30 rounds")
        break

Timing = time.process_time() - Timing
if Iteration > 0:
    print("Time per training loop: ", Timing/Iteration, " seconds")

input("Press [enter] to EXIT")
sys.exit(0)

# END
###################################################################################################
