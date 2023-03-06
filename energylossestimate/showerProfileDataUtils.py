import numpy as np

def pickEvent(random, event_list, selection):
    event_to_analyze = event_list[selection(event_list)]
    if random:
        r = random.randint(0, len(event_list))
        event_to_analyze = event_list[r]
    return event_to_analyze

def toDataSpace(event):

    x_vals = []
    y_vals = []
    z_vals = []

    for hit in event.hits:
        x_vals.append(hit[0])
        y_vals.append(hit[1])
        z_vals.append(hit[2])

    x_vals, y_vals, z_vals = map(np.array, [x_vals, y_vals, z_vals])
    D = np.column_stack((x_vals, y_vals, z_vals))

    """
       |       |
       | | | | |
       | x y z |
       | | | | |
       |       |
    """

    return D