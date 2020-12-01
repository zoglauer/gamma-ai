###################################################################################################
#
# GraphRepresentation.py
#
# Copyright (C) by Pranav Nagarajan & Rithwik Sudharsan
# All rights reserved.
#
# Please see the file LICENSE in the main repository for the copyright-notice.
#
###################################################################################################

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import random

from PIL import Image


# Class for the graph representation for the detector

# Default Values:
radius_default = 25
visualization_threshold = 0.5

class GraphRepresentation:

    # Map of all graph representations, indexed by EventID
    allGraphs = {}

    # Parameters:
    # Radius: Criterion for choosing to connect two nodes
    # Event: all event data to be used for this graph
    # pad_size: Define dimensions of input data

    ########
    # NOTE #
    ########
    # Do not use this to initialize graph, use GraphRepresentation.newGraphRepresentation
    def __init__(self, event, radius=radius_default, threshold=visualization_threshold):

        # Checking if distance is within criterion
        def DistanceCheck(h1, h2):
            dist = np.sqrt(np.sum((h1 - h2) ** 2))
            return dist <= radius

        A = np.zeros((len(event.X), len(event.X)))

        # Parse the event data
        assert len(event.X) == len(event.Y) \
               == len(event.Z) == len(event.E) \
               == len(event.Type) == len(event.Origin), "Event Data size mismatch."
        data = np.array(list(zip(event.X, event.Y, event.Z, event.E, event.Type, event.Origin)))
        hits = data[:, :3].astype(np.float32)
        energies = data[:, 3].astype(np.float32)
        types = data[:, 4]
        origins = data[:, 5].astype(np.int)

        # Note: how can gamma_bool or compton_bool be calculated beforehand
        # when evaluating on test data?

        # Fill in the adjacency matrix
        for i in range(len(hits)):
            for j in range(i + 1, len(hits)):
                gamma_bool = (types[i] == 'g' and types[j] == 'g')
                compton_bool = (types[i] == 'eg' or types[j] == 'eg')
                if compton_bool or gamma_bool or DistanceCheck(hits[i], hits[j]):
                    A[i][j] = A[j][i] = 1

        # Note: Ro and Ri are technically twice as large as necessary,
        # since the number of edges already indicates half a number of edges that can never be incoming.

        # Create the incoming matrix, outgoing matrix, and matrix of labels
        num_edges = int(np.sum(A))
        Ro = np.zeros((len(hits), num_edges), dtype = np.float32)
        Ri = np.zeros((len(hits), num_edges), dtype = np.float32)
        y = np.zeros(num_edges, dtype = np.float32)
        y_adj = np.zeros((len(hits), len(hits)))
        compton_arr = np.zeros(num_edges)
        type_arr = np.empty(num_edges, dtype = "S4")

        # Fill in the incoming matrix, outgoing matrix, and matrix of labels
        counter = 0
        for i in range(len(A)):
            for j in range(len(A[0])):
                if A[i][j]:
                    Ro[i, counter] = 1
                    Ri[j, counter] = 1
                    if i + 1 == origins[j]:
                        y_adj[i][j] = 1
                        y[counter] = 1
                        if types[i] == 'eg':
                            compton_arr[counter] = 1
                        type_arr[counter] = types[i] + types[j]
                    counter += 1

        # Generate feature matrix of nodes
        X = data[:, :4].astype(np.float32)

        # Visualize true edges of graph
        # VisualizeGraph(y_adj)

        self.graphData = [A, Ro, Ri, X, y]
        self.trueAdjMatrix = y_adj
        self.XYZ = hits
        self.EventID = event.EventID
        self.E = event.E
        self.Type = event.Type
        self.Origin = event.Origin
        self.Compton = compton_arr
        self.Tracks = type_arr

        # Stores all predicted adjacency matrices
        ########
        # NOTE #
        ########
        # Might create unnecessary data to be stored, but allows for post-training analysis
        self.predictedAdjMatrices = []
        ########
        # NOTE #
        ########
        # Creates many graphvisualizer objects (1 for each event) when technically only 1 is needed?
        self.threshold = threshold

        # Add this graph to the map of all graph representations
        GraphRepresentation.allGraphs[self.EventID] = self

    @staticmethod
    def newGraphRepresentation(event, radius=radius_default, threshold=visualization_threshold):
        # Returns the graph representation of the current event if it already exists, otherwise creates a new one.
        if event.EventID in GraphRepresentation.allGraphs:
            return GraphRepresentation.allGraphs[event.EventID]
        else:
            return GraphRepresentation(event, radius=radius, threshold=threshold)

    def save_graph(self, lastAdjMatrix, file):
        dimension = 'both'
        if dimension == 'both' or dimension == 'XZ':
            plt.figure(1)
            self.draw_hits(self.trueAdjMatrix, 'XZ')
            plt.figure(2)
            self.draw_hits(lastAdjMatrix, 'XZ')
            plt.savefig(file, format="PNG")
        if dimension == 'both' or dimension == 'YZ':
            plt.figure(1)
            self.draw_hits(self.trueAdjMatrix, 'YZ')
            plt.figure(2)
            self.draw_hits(lastAdjMatrix, 'YZ')
            plt.savefig(file, format="PNG")
        plt.close()
        return

    @staticmethod
    def saveAllGraphs(resultDir):
        # inefficient, runs through everything in 2N time instead of N time, but
        # its nice because we can print the id's that are going to be saved.
        ids = [id for id in list(GraphRepresentation.allGraphs.keys())
               if len(GraphRepresentation.allGraphs[id].predictedAdjMatrices) > 0]
        print("Initiate Visualizations: ID's {} to {}".format(ids[0], ids[-1]))

        # We don't save all graphs, only NUM_GRAPHS.
        # RANDOM_SAMPLE indicates whether to take random graphs, or first N.
        RANDOM_SAMPLE = False
        NUM_GRAPHS = 5

        if RANDOM_SAMPLE:
            loop_iter = random.sample(ids, NUM_GRAPHS)
        else:
            loop_iter = ids[:NUM_GRAPHS]

        for id in loop_iter:
            print("Saving graph {}".format(id), end="\r")
            graph = GraphRepresentation.allGraphs[id]
            numPred = len(graph.predictedAdjMatrices)
            graph.save_graph(graph.trueAdjMatrix, resultDir + os.path.sep + "Graph_{}_True".format(id))
            images = []
            # also feels very inefficient; saving images, then reading them, then deleting them
            # but simplest way to make GIF without finding way to load plt.savefig directly into PIL.Image.
            os.mkdir(resultDir + os.path.sep + "Graph_{}".format(id))
            for i in range(numPred):
                adj = graph.predictedAdjMatrices[i]
                fname = resultDir + os.path.sep + "Graph_{}".format(id) + os.path.sep + "Pred_{}".format(i)
                graph.save_graph(adj, fname)
                images.append(Image.open(fname))

            # GIF making from PIL.Image.
            images[0].save(resultDir + os.path.sep + "Graph_{}_Predictions.gif".format(id),
                           save_all=True, append_images=images[1:], optimize=False, duration=40, loop=0)

            for i in range(numPred):
                fname = resultDir + os.path.sep + "Graph_{}_Pred_{}".format(id, i)
                #os.remove(fname)

            print("Saved!", end="\r")


    # Given a vector of edge existence probabilities,
    # converts to adjacency matrix and adds to the list predictedAdjMatrices
    def add_prediction(self, pred):
        #print(pred)
        #print(self.graphData[1][0])
        def ConvertToAdjacency(A, output):
            result = np.zeros((len(A), len(A[0])))
            assert result.shape == A.shape; "Bad shape in adj conversion."
            counter = 0
            for i in range(len(A)):
                for j in range(len(A[0])):
                    if A[i][j]:
                        result[i][j] = output[counter]
                        counter += 1
            return result
        self.predictedAdjMatrices.append(ConvertToAdjacency(self.graphData[0], pred))


    # Shows correct graph representation (from simulation)
    # AND last prediction, for comparison. Shows in both XZ and YZ projections.
    # Parameters:
    # dimension: 'XZ', 'YZ', or 'both', to denote which two dimensions to project on
    # close_time: any int/float, or 'DO NOT CLOSE' to denote time before closing the visualization window
    def visualize_last_prediction(self, dimension='both', close_time="DO NOT CLOSE"):
        lastAdjMatrix = self.predictedAdjMatrices[-1]
        if dimension == 'both' or dimension == 'XZ':
            plt.figure(1)
            self.draw_hits(self.trueAdjMatrix, 'XZ')
            plt.figure(2)
            self.draw_hits(lastAdjMatrix, 'XZ')
            plt.show()
        if dimension == 'both' or dimension == 'YZ':
            plt.figure(1)
            self.draw_hits(self.trueAdjMatrix, 'YZ')
            plt.figure(2)
            self.draw_hits(lastAdjMatrix, 'YZ')
            plt.show()
        if close_time != "DO NOT CLOSE":
            plt.pause(close_time)
            plt.close()
        return

    # print(instance of GraphRepresentation) illustrates graph representation of correct hits
    def __str__(self):
        self.draw_hits(4)
        return "Graph Representation and Data for EventID = {}".format(self.EventID)

    # Parameters:
    # adj_matrix: adjacency matrix with probabilities of true edge.
    # -- Default value is a placeholder 0 to denote event.trueAdjMatrix, defined within the method.
    # dimensions: whether to project onto XZ plane or YZ plane.
    def draw_hits(self, adjmatrix, dimensions='XZ'):
        adjmatrix = np.where(adjmatrix > self.threshold, adjmatrix, 0)
        edge_colors = [c for c in (adjmatrix.flatten(order='C') * 100).tolist() if c != 0]
        positions = self.XYZ
        types = self.Type
        assert adjmatrix.shape[0] == adjmatrix.shape[1] == positions.shape[0]
        # The above line means adjacency matrix needs rows/columns even for nodes that have no edges at all.
        position_map = {}
        node_colors = [0 for _ in range(adjmatrix.shape[0])]
        for i in range(adjmatrix.shape[0]):
            position_map[i] = [positions[i, 0], positions[i, 2]] if dimensions == 'XZ' \
                else [positions[i, 1], positions[i, 2]]
            if types[i] != "e":
                node_colors[i] = 20

        G = nx.from_numpy_matrix(adjmatrix, create_using=nx.DiGraph)

        nx.draw_networkx(G=G, pos=position_map, arrows=True, with_labels=True, node_color=node_colors,
                         edge_color=edge_colors, edge_cmap=plt.get_cmap('cool'), edge_vmin=self.threshold*100, edge_vmax=100)

        return

'''
from EventData import EventData

data = EventData()
data.createFromToyModel(0)
rep = GraphRepresentation(data)
mat = np.zeros((len(rep.graphData[1][0]))) + 0.75

rep.add_prediction(mat)
print(rep.predictedAdjMatrices[0])
rep.visualize_last_prediction()
'''
