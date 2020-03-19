# Rithwik Sudharsan

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


''' Example graph plotting: K_3,3 (3 by 3 connected bipartite graph) with directed edges  '''
row1 = [0,0,0,0,0,0] # used for the first 3 rows of the K_3,3 adjacency matrix
row2 = [1,1,1,0,0,0] # used for the last 3 rows of the K_3,3 adjacency matrix
# Standard adjacency matrix (this one is of K_3,3)
# Directed edges from column label node to row label node
# (i.e. if element (4,1) = 1, then
adjacency=np.matrix([row1, row1, row1, row2, row2, row2])
# Dictionary with keys as node labels (same ordering as adjacency matrix)
# and values as 2D positions.
nodes = {0: [0, 1], 1: [1, 1], 2: [2, 1], 3: [0, 0], 4: [1, 0], 5: [2, 0]}
# Generating the graph
G=nx.from_numpy_matrix(adjacency, create_using=nx.DiGraph)
# Drawing the graph (must supply graph AND node positions)
nx.draw_networkx(G=G, pos=nodes, arrows=True, with_labels=True, )
# Show
plt.show()

# There are ways to include other labels
# (such as particle energy, ordering by index of hits, etc.)
# but for now the adjacency matrix must be in the same order as the node
# positions.
