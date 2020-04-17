###################################################################################################
#
# ComptonTrackIdentificationGNN_PyTorch.py
#
# Copyright (C) by Andreas Zoglauer & Pranav Nagarajan
# All rights reserved.
#
# Please see the file LICENSE in the main repository for the copyright-notice.
#
###################################################################################################

import numpy as np
from GraphVisualizer import GraphVisualizer


# Class for the graph representation for the detector

# Default Values:
pad_size_default = 100
radius_default = 25
visualization_threshold = 0.5

class GraphRepresentation:
    # Parameters:
    # Radius: Criterion for choosing to connect two nodes
    # Event: all event data to be used for this graph
    # pad_size: Define dimensions of input data

    def __init__(self, event, pad_size=pad_size_default, radius=radius_default):

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
        hits = data[:, :3].astype(np.float)
        energies = data[:, 3].astype(np.float)
        types = data[:, 4]
        origins = data[:, 5].astype(np.int)

        # Fill in the adjacency matrix
        for i in range(len(hits)):
            for j in range( i +1, len(hits)):
                gamma_bool = (types[i] == 'g' and types[j] == 'g')
                compton_bool = (types[j] == 'eg' and origins[j] == 1)
                if gamma_bool or compton_bool or DistanceCheck(hits[i], hits[j]):
                    A[i][j] = A[j][i] = 1

        # Create the incoming matrix, outgoing matrix, and matrix of labels
        num_edges = int(np.sum(A))
        Ro = np.zeros((len(hits), num_edges))
        Ri = np.zeros((len(hits), num_edges))
        y = np.zeros(pad_size)
        y_adj = np.zeros((len(hits), len(hits)))

        # Fill in the incoming matrix, outgoing matrix, and matrix of labels
        counter = 0
        for i in range(len(A)):
            for j in range(len(A[0])):
                if A[i][j]:
                    Ro[i, np.arange(num_edges)] = 1
                    Ri[j, np.arange(num_edges)] = 1
                    if i + 1 == origins[j]:
                        y_adj[i][j] = 1
                        y[counter] = 1
                    counter += 1

        # Generate feature matrix of nodes
        X = data[:, :4].astype(np.float)

        # Visualize true edges of graph
        # VisualizeGraph(y_adj)

        self.graphData = [A, Ro, Ri, X, y]
        self.hitXYZ = hits
        self.trueAdjMatrix = y_adj

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
        self.visualizer = GraphVisualizer(self, visualization_threshold)


    def add_prediction(self, pred):

        def ConvertToAdjacency(A, output):
            result = np.zeros(len(A), len(A[0]))
            counter = 0
            for i in range(len(A)):
                for j in range(len(A[0])):
                    if A[i][j]:
                        result[i][j] = output[counter]
                        counter += 1
            return result

        self.predictedAdjMatrices.append(ConvertToAdjacency(self.trueAdjMatrix, pred))

    # Shows correct graph representation AND last prediction, for comparison. Shows in both XZ and YZ projections.
    def visualize_last_prediction(self, close_time=5):
        lastAdjMatrix = self.predictedAdjMatrices[len(self.predictedAdjMatrices) - 1]
        self.visualizer.Visualize_Hits(0, 'XZ')
        self.visualizer.Visualize_Hits(lastAdjMatrix, 'XZ')
        self.visualizer.Visualize_Hits(0, 'YZ')
        self.visualizer.Visualize_Hits(lastAdjMatrix, 'YZ')

        self.visualizer.closeVisualization(close_time)
        return