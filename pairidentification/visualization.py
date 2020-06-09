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
    plt.plot(x_coords, y_coords, color)[0]


def draw_edge_xyz(ptA, ptB, color="blue"):
    x_coords = np.array([ptA[0], ptB[0]])
    y_coords = np.array([ptA[1], ptB[1]])
    z_coords = np.array([ptA[2], ptB[2]])
    plt.plot(x_coords, y_coords, z_coords, color)[0]
    

def draw_vector_2d(ptA, ptB, color="blue"):
    plt.quiver(
        ptA[0], ptA[1], # start point
        ptB[0] - ptA[0], ptB[1]-ptA[1], # direction
        scale=1, angles='xy', scale_units='xy', color=color
    )


def draw_vector_3d(ptA, ptB, color="blue"):
    plt.quiver(
        ptA[0], ptA[1], ptA[2], # start point
        ptB[0] - ptA[0], ptB[1]-ptA[1], ptB[2]-ptA[2], # direction
        color=color,
        arrow_length_ratio=0.05,
    )


def filter_position(pos):
    """
    Filters out padded rows
    """
    pos = np.array(pos)
    new_pos = []
    for i in range(len(pos)):
        row = pos[i]
        print
        if not np.any(row):
            if i == len(pos) - 1: # last point
                break
            elif not np.any(pos[i+1]): # next point is also zero
                break
        new_pos.append(row)
    
    new_pos = np.array(new_pos)
    return new_pos


"""
pos is (max_hits, 3) containing XYZ
Rin, Rout is (max_hits, max_edges)
predicted_edges, generated_edges is (1, max_edges), True if actually present
axis is tuple of the 2 axis to plot
"""
def draw_2d_plot(pos, Rin, Rout, predicted_edges, generated_edges, True_Ri, True_Ro, axis=(0, 1)):
    fig = plt.figure(figsize=(9, 9))
    new_pos = filter_position(pos)
    plt.scatter(new_pos[:, axis[0]], new_pos[:, axis[1]])
    
    num_edges = Rin.shape[1]
    for edge_idx in range(num_edges):
        
        # checking if edge or padding
        if sum(Rin[:, edge_idx]) != 0:
            ptA_idx = np.nonzero(Rout[:, edge_idx])[0][0]
            ptB_idx = np.nonzero(Rin[:, edge_idx])[0][0]
            
            ptA = np.array([pos[ptA_idx][axis[0]], pos[ptA_idx][axis[1]]])
            ptB = np.array([pos[ptB_idx][axis[0]], pos[ptB_idx][axis[1]]])
            
            color = edge_color(edge_idx, predicted_edges, generated_edges)
            if color is not None:
#                 draw_edge_xy(ptA, ptB, color)
                draw_vector_2d(ptA, ptB, color)
    
    # Script error
    pt_indices = compare_True_Manual_edges(Rin, Rout, True_Ri, True_Ro)
    for pair in pt_indices:
        ptA = pos[pair[0]]
        pointA = [ptA[axis[0]], ptA[axis[1]]]
        ptB = pos[pair[1]]
        pointB = [ptB[axis[0]], ptB[axis[1]]]
#         draw_edge_xy(pointA, pointB, "orange")
        draw_vector_2d(pointA, pointB, "orange")
    
    axis_dictionary = {0:'x', 1:'y', 2:'z'}
    x_label = axis_dictionary[axis[0]]
    y_label = axis_dictionary[axis[1]]
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(x_label+y_label+" plot")

#     plt.savefig(OutputDirectory +'/test.png')

"""
pos is (max_hits, 3) containing XYZ
Rin, Rout is (max_hits, max_edges)
predicted_edges, generated_edges is (1, max_edges), True if actually present
"""
def draw_3d_plot(pos, Rin, Rout, predicted_edges, generated_edges, True_Ri, True_Ro):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    new_pos = filter_position(pos)
    ax.scatter(new_pos[:, 0], new_pos[:, 1], new_pos[:, 2])
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


def draw_3d_arrows(pos, Rin, Rout, predicted_edges, generated_edges, True_Ri, True_Ro):
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
                draw_vector_3d(ptA, ptB, color)
#                 ax.annotate("", xy=ptB, xytext=ptB-ptA, arrowprops=dict(arrowstyle="->"))

    
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


class GraphVisualizer(object):
    axis_dictionary = {0:'x', 1:'y', 2:'z'}
    
    def __init__(self, summary, labels, True_Ri, True_Ro, OutputDir, figure_size=(9, 9)):
        self.summary = summary
        self.labels = labels
        self.True_Ri = True_Ri
        self.True_Ro = True_Ro
        self.OutputDir = OutputDir
        self.figure_size = figure_size
        
        # default batch index
        self.batch_idx = len(summary['X']) - 1


    def draw_2d(self, sample_idx, batch_idx=None, axis=(0,1), filename=None, save=True):
        if batch_idx == None:
            batch_idx = self.batch_idx
        pos = np.array(self.summary['X'][batch_idx][sample_idx]) 
        
        # Might want to convert everything to numpy?
        Rin = self.summary['Ri'][batch_idx][sample_idx]
        Rout = self.summary['Ro'][batch_idx][sample_idx]
        predicted_edges = self.summary['Edge_Labels'][batch_idx][sample_idx]
        generated_edges = self.labels[sample_idx]
        True_Ri = self.True_Ri[sample_idx]
        True_Ro = self.True_Ro[sample_idx]
      
        fig = plt.figure(figsize=self.figure_size)
        
        # Filtering out padded rows, plotting points
        new_pos = filter_position(pos)
        plt.scatter(new_pos[:, axis[0]], new_pos[:, axis[1]])

        num_edges = Rin.shape[1]
        for edge_idx in range(num_edges):

            # checking if edge or padding
            if sum(Rin[:, edge_idx]) != 0:
                ptA_idx = np.nonzero(Rout[:, edge_idx])[0][0]
                ptB_idx = np.nonzero(Rin[:, edge_idx])[0][0]

                ptA = np.array([pos[ptA_idx][axis[0]], pos[ptA_idx][axis[1]]])
                ptB = np.array([pos[ptB_idx][axis[0]], pos[ptB_idx][axis[1]]])

                color = edge_color(edge_idx, predicted_edges, generated_edges)
                if color is not None:
                    # draw_edge_xy(ptA, ptB, color)
                    draw_vector_2d(ptA, ptB, color)

        # Script error
        pt_indices = compare_True_Manual_edges(Rin, Rout, True_Ri, True_Ro)
        for pair in pt_indices:
            ptA = pos[pair[0]]
            pointA = [ptA[axis[0]], ptA[axis[1]]]
            ptB = pos[pair[1]]
            pointB = [ptB[axis[0]], ptB[axis[1]]]
            # draw_edge_xy(pointA, pointB, "orange")
            draw_vector_2d(pointA, pointB, "orange")

        
        x_label = GraphVisualizer.axis_dictionary[axis[0]]
        y_label = GraphVisualizer.axis_dictionary[axis[1]]
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(x_label+y_label+" plot")
        
        # Saving plot
        if save:
            if filename == None:
                filename = "{}{}_plot_{}_{}".format(x_label, y_label, batch_idx, sample_idx)
            plt.savefig(self.OutputDir +'/' + filename)
    
    
    def draw_3d(self, sample_idx, batch_idx=None, filename=None, save=True, arrow=False):
        if batch_idx == None:
            batch_idx = self.batch_idx
        pos = np.array(self.summary['X'][batch_idx][sample_idx]) 
        
        # Might want to convert everything to numpy?
        Rin = self.summary['Ri'][batch_idx][sample_idx]
        Rout = self.summary['Ro'][batch_idx][sample_idx]
        predicted_edges = self.summary['Edge_Labels'][batch_idx][sample_idx]
        generated_edges = self.labels[sample_idx]
        True_Ri = self.True_Ri[sample_idx]
        True_Ro = self.True_Ro[sample_idx]
      
        fig = plt.figure(figsize=self.figure_size)
        ax = fig.add_subplot(111, projection='3d')
        
        new_pos = filter_position(pos)
        ax.scatter(new_pos[:, 0], new_pos[:, 1], new_pos[:, 2])
        
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
                    if arrow:
                        draw_vector_3d(ptA, ptB, color)
                        # ax.annotate("", xy=ptB, xytext=ptB-ptA, arrowprops=dict(arrowstyle="->"))
                    else:
                        draw_edge_xyz(ptA, ptB, color)


        # Script error
        pt_indices = compare_True_Manual_edges(Rin, Rout, True_Ri, True_Ro)
        for pair in pt_indices:
            ptA = pos[pair[0]]
            ptB = pos[pair[1]]
            if arrow:
                draw_vector_3d(ptA, ptB, "orange")
            else:
                draw_edge_xyz(ptA, ptB, "orange")
            
        # Saving plot
        if save:
            if filename == None:
                filename = "3d_plot_{}_{}".format(batch_idx, sample_idx)
            plt.savefig(self.OutputDir +'/' + filename)


    def plot_sample(self, sample_idx, batch_idx=None, save=True):
        if batch_idx == None:
            batch_idx = self.batch_idx
        self.draw_2d(sample_idx, batch_idx=batch_idx, axis=(0, 1), save=save)
        self.draw_2d(sample_idx, batch_idx=batch_idx, axis=(0, 2), save=save)
        self.draw_2d(sample_idx, batch_idx=batch_idx, axis=(1, 2), save=save)
        
        self.draw_3d(sample_idx, batch_idx=batch_idx, save=save)
    
    