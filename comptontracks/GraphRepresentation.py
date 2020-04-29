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

import numpy as np
from GraphVisualizer import GraphVisualizer


# Class for the graph representation for the detector

# Default Values:
pad_size_default = 100
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

        # Padding to maximum dimension
        A = np.pad(A, [(0, pad_size - len(A)), (0, pad_size - len(A[0]))])
        Ro = np.pad(Ro, [(0, pad_size - len(Ro)), (0, pad_size - len(Ro[0]))], constant_values = 2)
        Ri = np.pad(Ri, [(0, pad_size - len(Ri)), (0, pad_size - len(Ri[0]))], constant_values = 2)
        X = np.pad(X, [(0, pad_size - len(X)), (0, 0)])

        self.graphData = [A, Ro, Ri, X, y]
        self.trueAdjMatrix = y_adj
        self.XYZ = hits
        self.EventID = event.EventID
        self.E = event.E
        self.Type = event.Type
        self.Origin = event.Origin

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

        # Add this graph to the map of all graph representations
        GraphRepresentation.allGraphs[self.EventID] = self

    @staticmethod
    def newGraphRepresentation(event, pad_size=pad_size_default, radius=radius_default):
        # Returns the graph representation of the current event if it already exists, otherwise creates a new one.
        return GraphRepresentation.allGraphs.get(event.EventID, GraphRepresentation(event, pad_size, radius))

    # Given a vector of edge existence probabilities,
    # converts to adjacency matrix and adds to the list predictedAdjMatrices
    def add_prediction(self, pred):

        def ConvertToAdjacency(A, output):
            result = np.zeros((len(A), len(A[0])))
            counter = 0
            for i in range(len(A)):
                for j in range(len(A[0])):
                    if A[i][j]:
                        result[i][j] = output[counter]
                        counter += 1
            return result

        self.predictedAdjMatrices.append(ConvertToAdjacency(self.trueAdjMatrix, pred))

    # Shows correct graph representation (from simulation)
    # AND last prediction, for comparison. Shows in both XZ and YZ projections.
    # Parameters:
    # dimension: 'XZ', 'YZ', or 'both', to denote which two dimensions to project on
    # close_time: any int/float, or 'DO NOT CLOSE' to denote time before closing the visualization window
    def visualize_last_prediction(self, dimension='both', close_time=5):
        lastAdjMatrix = self.predictedAdjMatrices[len(self.predictedAdjMatrices) - 1]
        if dimension == 'both' or dimension == 'XZ':
            self.visualize_simulation('XZ', "DO NOT CLOSE")
            self.visualizer.visualize_hits(lastAdjMatrix, 'XZ')
        if dimension == 'both' or dimension == 'YZ':
            self.visualize_simulation('YZ', "DO NOT CLOSE")
            self.visualizer.visualize_hits(lastAdjMatrix, 'YZ')
        self.visualizer.closeVisualization(close_time)
        return

    # Visualize correct output only (from simulation)
    # Parameters:
    # dimension: 'XZ', 'YZ', or 'both', to denote which two dimensions to project on
    # close_time: any int/float, or 'DO NOT CLOSE' to denote time before closing the visualization window
    def visualize_simulation(self, dimension='both', close_time=5):
        if dimension == 'both' or dimension == 'XZ':
            self.visualizer.visualize_hits(0, 'XZ')
        if dimension == 'both' or dimension == 'YZ':
            self.visualizer.visualize_hits(0, 'YZ')
        if close_time != "DO NOT CLOSE":
            self.visualizer.closeVisualization(close_time)

    # print(instance of GraphRepresentation) illustrates graph representation of correct hits
    def __str__(self):
        self.visualizer.visualize_hits(4)
        return "Graph Representation and Data for EventID = {}".format(self.EventID)
