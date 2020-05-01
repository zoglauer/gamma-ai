import numpy as np
def generate_incidence(edges, pos_data):
    #Generate Incidence Matrix from Edge List
    n_hits = len(pos_data)
    n_edges = len(edges)
    Ri = np.zeros((n_hits, n_edges), dtype=np.uint8)
    Ro = np.zeros((n_hits, n_edges), dtype=np.uint8)
    
    for i in range(len(edges)):
        point = edges[i]
        from_pt = point[0]
        to_pt = point[1]
        Ro[from_pt][i] = 1
        Ri[to_pt][i] = 1
    
    return Ri, Ro

def connect_pos(pos_data):
    #Manually Connect Graph based on Positions
    edges = []
    for i in range(len(pos_data)):
        point_A = pos_data[i]
        z_A = point_A[2]

        for j in range(len(pos_data)):
            point_B = pos_data[j]
            z_B = point_B[2]

            if z_B == z_A + 1:
                edges.append((i, j))
                edges.append((j, i))
    return generate_incidence(edges, pos_data)

def pad(arr, shape):
    #Padd arr to Shape
    padded_arr = np.zeros(shape)
    if len(shape) == 1:
        padded_arr[:arr.shape[0]] = arr
    elif len(shape) == 2:
        padded_arr[:arr.shape[0],:arr.shape[1]] = arr
    return padded_arr

def vectorize_data(eventArr):
    # Edge Validity Labels, Manually Connected Rin, Mannually Connected Rout, XYZ, Type, Energy, Gamma Energy
    Edge_Labels, True_Ri, True_Ro, Man_Ri, Man_Ro, XYZ, Type, Energy, GammaEnergy = [], [], [], [], [], [], [], [], []
    max_hits, max_edges = 0, 0
    
    #Start Parsing Events
    for event in eventArr:
        #Keep track of max hits for padding
        max_hits = max(max_hits, len(event.X))
        
        #Generate Incidence Matrices based on Edges
        edges = []
        pos = np.swapaxes(np.vstack((event.X, event.Y, event.Z)), 0, 1)
        for i in range(1,len(event.Origin+1)):
            edges.append((event.Origin[i-1]-1,i-1))
        e_Ri, e_Ro = generate_incidence(edges,pos)
        
        #Generate Proposed Incidence Matrices based on Positions
        p_Ri, p_Ro = connect_pos(pos)
        
        #Generate Edge Labels (0 - fake edge; 1 - true edge)
        e_label = np.zeros(p_Ri.shape[1])
        for i in range(p_Ri.shape[1]):
            out = np.where(p_Ro[:,i] == 1)[0][0]
            inn = np.where(p_Ri[:,i] == 1)[0][0]
            e_label[i] = 1*((out, inn) in edges)
            
        #Keep track of max edges for Padding
        max_edges = max(max_edges, p_Ri.shape[1])
        
        #Add all of the event data to lists
        Edge_Labels.append(e_label)
        True_Ri.append(e_Ri)
        True_Ro.append(e_Ro)
        Man_Ri.append(p_Ri)
        Man_Ro.append(p_Ro)
        XYZ.append(np.vstack((event.X, event.Y, event.Z)).T)
        Type.append(2*(event.Type=='m')+(event.Type=='p'))
        Energy.append(event.E)
        GammaEnergy.append(event.GammaEnergy)
    
    #Padding based on Max Hits and Max Edges
    for i in range(len(Edge_Labels)):
        Edge_Labels[i] = pad(Edge_Labels[i],(max_edges,))
        Man_Ri[i] = pad(Man_Ri[i],(max_hits,max_edges))
        Man_Ro[i] = pad(Man_Ro[i],(max_hits,max_edges))
        XYZ[i] = pad(XYZ[i],(max_hits,3))
        Type[i] = pad(Type[i],(max_hits,))
        Energy[i] = pad(Energy[i],(max_hits,))
    
    return np.array(Edge_Labels, dtype=np.float32), np.array(Man_Ri, dtype=np.float32), np.array(Man_Ro, dtype=np.float32), np.array(XYZ, dtype=np.float32), np.array(Type, dtype=np.float32), np.array(Energy, dtype=np.float32), np.array(GammaEnergy, dtype=np.float32), True_Ri, True_Ro

def generate_dataset(TrainingDataSets):
    Edge_Labels, Man_Ri, Man_Ro, XYZ, Type, Energy, GammaEnergy, True_Ri, True_Ro = vectorize_data(TrainingDataSets)
    features = [[XYZ[i], Man_Ri[i], Man_Ro[i]] for i in range(XYZ.shape[0])]
    labels = Edge_Labels
    dataset = [[features[i],labels[i]] for i in range(XYZ.shape[0])]
    return dataset, Edge_Labels, True_Ri, True_Ro