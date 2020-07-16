###################################################################################################
#
# GraphVisualizer.py
#
# Copyright (C) by Rithwik Sudharsan
# All rights reserved.
#
# Please see the file LICENSE in the main repository for the copyright-notice.
#
###################################################################################################

### THIS FILE IS DEPRECATED ###
### Visualization is now done in GraphRepresentation.py ###

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

'''
row1 = [0,0,0,0,0,0] # used for the first 3 rows of the K_3,3 adjacency matrix
row2 = [0.3,1,0.3,0,0,0] # used for the last 3 rows of the K_3,3 adjacency matrix
# Standard adjacency matrix (this one is of K_3,3)
# Directed edges from column label node to row label node
# (i.e. if element (4,1) = 1, then
adjacency=np.matrix([row1, row1, row1, row2, row2, row2])
# Dictionary with keys as node labels (same ordering as adjacency matrix)
# and values as 2D positions.
nodes = {0: [0, 1], 1: [1, 1], 2: [2, 1], 3: [0, 0], 4: [1, 0], 5: [2, 0]}
# Generating the graph
G=nx.from_numpy_matrix(adjacency, create_using=nx.DiGraph)
colors = [20 for i in range(9)]
colors[0]=100
# Drawing the graph (must supply graph AND node positions)
plt.figure(1)
nx.draw_networkx(G=G, pos=nodes, arrows=True, with_labels=True, node_size = 50, edge_color=colors, edge_cmap=plt.get_cmap('RdYlGn'))
# Show
plt.figure(2)
nx.draw_networkx(G=G, pos=nodes, arrows=True, with_labels=True, node_size = 50, edge_color=colors, edge_cmap=plt.get_cmap('Accent'))
plt.show()
'''


# https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.drawing.nx_pylab.draw_networkx.html
# https://stackoverflow.com/questions/25639169/networkx-change-color-width-according-to-edge-attributes-inconsistent-result

# There are ways to include other labels
# (such as particle energy, ordering by index of hits, etc)
# but for now the adjacency matrix must be in the same order as the node
# positions. The node positions have to be 2D, so maybe use the positions
# as projected on the Y-Z plane or X-Z plane (ignore either Y coordinate or X coordinate).

class GraphVisualizer:
    # Parameters:
    # threshold = cutoff probability for counting a hit as valid or invalid.
    # graphrep: GraphRepresentation object that holds all graph data.
    def __init__(self, graphrep, threshold=0.5):
        self.threshold = threshold
        self.graphrep = graphrep

    # Parameters:
    # adj_matrix: adjacency matrix with probabilities of true edge.
    # -- Default value is a placeholder 0 to denote event.trueAdjMatrix, defined within the method.
    # dimensions: whether to project onto XZ plane or YZ plane.
    def visualize_hits(self, adjmatrix, dimensions='XZ'):
        positions = self.graphrep.XYZ
        types = self.graphrep.Type
        assert adjmatrix.shape[0] == adjmatrix.shape[1] == positions.shape[0]
        # The above line means adjacency matrix needs rows/columns even for nodes that have no edges at all.
        position_map = {}
        colors = [0 for _ in range(adjmatrix.shape[0])]
        for i in range(adjmatrix.shape[0]):
            position_map[i] = [positions[i, 0], positions[i, 2]] if dimensions == 'XZ' \
                else [positions[i, 1], positions[i, 2]]
            if types[i] != "e":
                colors[i] = 100
        G = nx.from_numpy_matrix(np.where(adjmatrix > self.threshold, 1, 0), create_using=nx.DiGraph)
        ########
        # NOTE #
        ########
        # Need to reorder nodes according to the scattering for node labels (with_labels=True) to actually be accurate!
        nx.draw_networkx(G=G, pos=position_map, arrows=True, with_labels=True, node_color=colors)
        plt.show()

        return

    # hit visualizer with variable darkness edges
    def visualize_hits_advanced(self, adjmatrix, dimensions='XZ'):
        adjmatrix = np.where(adjmatrix > self.threshold, adjmatrix, 0)
        edge_colors = [c for c in (adjmatrix.flatten(order='C') * 100).tolist() if c != 0]
        positions = self.graphrep.XYZ
        types = self.graphrep.Type
        assert adjmatrix.shape[0] == adjmatrix.shape[1] == positions.shape[0]
        # The above line means adjacency matrix needs rows/columns even for nodes that have no edges at all.
        position_map = {}
        colors = [0 for _ in range(adjmatrix.shape[0])]
        for i in range(adjmatrix.shape[0]):
            position_map[i] = [positions[i, 0], positions[i, 2]] if dimensions == 'XZ' \
                else [positions[i, 1], positions[i, 2]]
            if types[i] != "e":
                colors[i] = 100

        G = nx.from_numpy_matrix(adjmatrix, create_using=nx.DiGraph)
        ########
        # NOTE #
        ########
        # Need to reorder nodes according to the scattering for node labels (with_labels=True) to actually be accurate!
        nx.draw_networkx(G=G, pos=position_map, arrows=True, with_labels=True, node_color=colors,
                         edge_color=edge_colors, edge_cmap=plt.get_cmap('RdYlGn'), edge_vmin=self.threshold*100, edge_vmax=101)
        plt.show()

        return


    # Parameters:
    # time: number of seconds before closing all visualization windows.
    def closeVisualization(self, time=5):
        plt.pause(time)
        plt.close()


