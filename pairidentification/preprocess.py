def generate_incidence(edges, pos_data):
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
    print(edges)
    
    return generate_incidence(edges, pos_data)

def vectorize_data(eventArr):
    Ri, Ro = [], []
    xyz = []
    t = []
    E = []
    GE = []
    
    max_hits = 0
    max_edges = 0
    
    #parse events
    for event in eventArr:
        edges = []
        max_hits = max(max_hits, len(event.X))
        
        pos = np.swapaxes(np.vstack((event.X, event.Y, event.Z)), 0, 1)
        for i in range(1,len(event.Origin+1)):
            edges.append((i-1,event.Origin[i-1]-1))
        
        max_edges = max(max_edges, len(edges))
        
        e_Ri, e_Ro = generate_incidence(edges,pos)
        
        Ri.append(e_Ri)
        Ro.append(e_Ro)
        xyz.append(np.hstack((event.X, event.Y, event.Z)))
        t.append(2*(event.Type=='m')+(event.Type=='p'))
        E.append(event.E)
        GE.append(event.GammaEnergy)
    
    #padding
    for i in range(len(Ri)):
        arr = Ri[i]
        padded_arr = np.zeros((max_hits,max_edges))
        padded_arr[:arr.shape[0],:arr.shape[1]] = arr
        Ri[i] = padded_arr
        
        arr = Ro[i]
        padded_arr = np.zeros((max_hits,max_edges))
        padded_arr[:arr.shape[0],:arr.shape[1]] = arr
        Ro[i] = padded_arr
        
        arr = xyz[i]
        padded_arr = np.zeros((max_hits*3))
        padded_arr[:arr.shape[0]] = arr
        xyz[i] = padded_arr
        
        arr = t[i]
        padded_arr = np.zeros((max_hits))
        padded_arr[:arr.shape[0]] = arr
        t[i] = padded_arr
        
        arr = E[i]
        padded_arr = np.zeros((max_hits))
        padded_arr[:arr.shape[0]] = arr
        E[i] = padded_arr
    
    return np.array(Ri), np.array(Ro), np.array(xyz), np.array(t), np.array(E), np.array(GE)