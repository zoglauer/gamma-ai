import matplotlib.pyplot as plt
import numpy as np

def add_arrow(line, position=None, direction='right', size=15, color=None):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="->", color=color),
        size=size
    )


def draw_vertices_xy(points):
    plt.plot(points[:, 0], points[:, 1], 'o', c='black')

def draw_edge_xy(ptA, ptB, color="blue"):
    x_coords = np.array([ptA[0], ptB[0]])
    y_coords = np.array([ptA[1], ptB[1]])
    line = plt.plot(x_coords, y_coords, color)[0]
#     add_arrow(line)
#     if correct:
#         line = plt.plot(x_coords, y_coords, 'green')[0]
#         # add_arrow(line)
#     else:
#         line = plt.plot(x_coords, y_coords, 'red')[0]
#         # add_arrow(line)

def draw_edge_xyz(ptA, ptB, color="blue"):
    x_coords = np.array([ptA[0], ptB[0]])
    y_coords = np.array([ptA[1], ptB[1]])
    z_coords = np.array([ptA[2], ptB[2]])

    line = plt.plot(x_coords, y_coords, z_coords, color)[0]
#     if correct:
#         line = plt.plot(x_coords, y_coords, z_coords, 'green')[0]
# #         add_arrow(line)
#     else:
#         line = plt.plot(x_coords, y_coords, z_coords, 'red')[0]
# #         add_arrow(line)


"""
pos is (max_hits, 3) containing XYZ
Rin, Rout is (max_hits, max_edges)
predicted_edges, generated_edges is (1, max_edges), True if actually present
axis is tuple of the 2 axis to plot
"""
def draw_2d_plot(pos, Rin, Rout, predicted_edges, generated_edges, True_Ri, True_Ro, axis=(0, 1)):
    fig = plt.figure()
    plt.scatter(pos[:, axis[0]], pos[:, axis[1]])
    num_edges = Rin.shape[1]
    for edge_idx in range(num_edges):
        
        # checking if edge or padding
        if sum(Rin[:, edge_idx]) != 0:
            ptA_idx = np.nonzero(Rout[:, edge_idx])[0][0]
            ptB_idx = np.nonzero(Rin[:, edge_idx])[0][0]
            
            ptA = [pos[ptA_idx][axis[0]], pos[ptA_idx][axis[1]]]
            ptB = [pos[ptB_idx][axis[0]], pos[ptB_idx][axis[1]]]
            
            color = edge_color(edge_idx, predicted_edges, generated_edges)
            if color is not None:
                draw_edge_xy(ptA, ptB, color)
    
    # Script error
    pt_indices = compare_True_Manual_edges(Rin, Rout, True_Ri, True_Ro)
    for pair in pt_indices:
        ptA = pos[pair[0]]
        pointA = [ptA[axis[0]], ptA[axis[1]]]
        ptB = pos[pair[1]]
        pointB = [ptB[axis[0]], ptB[axis[1]]]
        draw_edge_xy(pointA, pointB, "orange")

"""
pos is (max_hits, 3) containing XYZ
Rin, Rout is (max_hits, max_edges)
predicted_edges, generated_edges is (1, max_edges), True if actually present
"""
def draw_3d_plot(pos, Rin, Rout, predicted_edges, generated_edges, True_Ri, True_Ro):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2])
    num_edges = Rin.shape[1]
    for edge_idx in range(num_edges):
        
        # checking if edge or padding
        if sum(Rin[:, edge_idx]) != 0:
            ptA_idx = np.nonzero(Rout[:, edge_idx])[0][0]
            ptB_idx = np.nonzero(Rin[:, edge_idx])[0][0]
            
            ptA = pos[ptA_idx]
            ptB = pos[ptB_idx]

            color = edge_color(edge_idx, predicted_edges, generated_edges)
            if color is not None:
                draw_edge_xyz(ptA, ptB, color)
    
    # Script error
    pt_indices = compare_True_Manual_edges(Rin, Rout, True_Ri, True_Ro)
    for pair in pt_indices:
        ptA = pos[pair[0]]
        ptB = pos[pair[1]]
        draw_edge_xyz(ptA, ptB, "orange")


'''
for edge in edge_labels
    if predicted true and it's in the event
        color edge green
    if predicted true and it's not in the event
        color edge red
    if predicted false and it's in the event
        color edge purple
    if predicted false and it's not in the event
        don't draw the edge
script error:
an edge in the event is not caught by the manual connection script
look through the edges in event
    if an edge is in the event and not in the manually connected graph -
        color edge orange
'''
def edge_color(edge_idx, predicted_edges, generated_edges):
    if predicted_edges[edge_idx] == True:
        if generated_edges[edge_idx] == True:
            color = 'green'
        else:
            color = 'red'
    else:
        if generated_edges[edge_idx] == True:
            color = 'purple'
        else:
            color = None
    return color


def compare_True_Manual_edges(Man_Ri, Man_Ro, True_Ri, True_Ro):
    pt_indices = []
    for true_edge_idx in range(True_Ri.shape[1]):
        ptA_idx = np.nonzero(True_Ro[:, true_edge_idx])[0][0]
        ptB_idx = np.nonzero(True_Ri[:, true_edge_idx])[0][0]
        
        edge_found = False
        
        A_row = Man_Ro[ptA_idx]
        for e_idx in range(len(A_row)):
            if A_row[e_idx] == 1:
                if Man_Ri[ptB_idx][e_idx] == 1:
                    edge_found = True
                    break
        
        if not edge_found:
            pt_indices.append([ptA_idx, ptB_idx])
    
    return pt_indices