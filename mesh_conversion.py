# mesh_conversion
import numpy as np


def validate_partition_and_get_interval_num(partition: np.array) -> int:
    num_interval  = len(partition) - 1
    for k in range(num_interval):
        if partition[k+1] <= partition[k]:
            raise ValueError("the given partition is not valid")
    return num_interval

def augment_values(values: np.array, length: int) -> (np.array, int):
    new_length = length*2
    new_values = np.zeros(new_length-1)
    new_values[0]  = values[0]/2
    new_values[-1] = values[-1]/2
    for k in range(1, new_length-2):
        if k%2 == 1:
            new_values[k] = values[k//2]
        if k%2 == 0:
            new_values[k] = (values[(k-1)//2] + values[(k+1)//2])/2

    return new_values, new_length

def augment_mesh(nodes: np.array, values: np.array, length: int) -> (np.array, np.array, int):
    new_length = length*2

    new_nodes  = np.array([nodes[k//2] if k%2 == 0 
                           else (nodes[(k-1)//2]+nodes[(k+1)//2])/2 
                           for k in range(new_length+1)])

    new_values = np.zeros(new_length-1)
    new_values[0]  = values[0]/2
    new_values[-1] = values[-1]/2
    for k in range(1, new_length-2):
        if k%2 == 1:
            new_values[k] = values[k//2]
        if k%2 == 0:
            new_values[k] = (values[(k-1)//2] + values[(k+1)//2])/2

    return new_nodes, new_values, new_length

def augmentation(nodes: np.array, values: np.array, target_size: int) -> (np.array, np.array):
    curr_size = validate_partition_and_get_interval_num(nodes)
    while curr_size < target_size:
        nodes, values, curr_size = augment_mesh(nodes, values, curr_size)
    return nodes, values, curr_size


def downsize_values(values: np.array,
                    length: int) -> np.array and int:
    new_length = length//2
    if new_length > 2:
        new_values = np.array([(values[k-1] + values[k]*2 + values[k+1])/2
                               for k in range(length-1) if k%2==1])
    else:
        new_values = (values[0] + values[1]*2 + values[2])/2
    return new_values, new_length

def downsize_mesh(nodes: np.array, values: np.array, length: int) -> (np.array, np.array, int):
    new_length = length//2

    new_nodes  = np.array([nodes[k] for k in range(length+1) if k%2==0 ])
    if new_length > 2:
        new_values = np.array([(values[k-1] + values[k]*2 + values[k+1])/2
                               for k in range(length-1) if k%2==1])
    else:
        new_values = (values[0] + values[1]*2 + values[2])/2
    return new_nodes, new_values, new_length

def downsizing(nodes: np.array, values: np.array, target_size: int) -> (np.array, np.array):
    curr_size = validate_partition_and_get_interval_num(nodes)
    while curr_size > target_size and curr_size > 2 and curr_size%2==0:
        nodes, values, curr_size = downsize_mesh(nodes, values, curr_size)
    return nodes, values, curr_size

def test_mesh_operations(num_nodes: int) -> None:
    nodes     = np.linspace(0, 1, num_nodes+2)
    values    = np.array([(-0.9)**k for k in range(num_nodes)])

    print("initial set-up")
    print("nodes: ", nodes)
    print("values:", values)

    augm_nodes, augm_values, length = augment_mesh(nodes, values, num_nodes)
    print("\nafter augmenting")
    print("nodes: ", augm_nodes)
    print("values:", augm_values)

    down_nodes, down_values, length = downsize_mesh(nodes, values, num_nodes)
    print("\nafter downsizing")
    print("nodes: ", down_nodes)
    print("values:", down_values)


def test_mesh_shortcuts(num_intervals: int) -> None:
    nodes  = np.linspace(0, 1, num_intervals+1)
    values = np.array([(-0.9)**k for k in range(num_intervals-1)])

    print("initial set-up")
    print("nodes: ", nodes)
    print("values:", values)

    augm_nodes, augm_values, augm_size = augmentation(nodes, values, 16)
    print("\nafter augmenting")
    print("nodes: ", augm_nodes)
    print("values:", augm_values)

    down_nodes, down_values, down_size = downsizing(nodes, values, 4)
    print("\nafter downsizing")
    print("nodes: ", down_nodes)
    print("values:", down_values)


if __name__ == "__main__":
    interval_number = 8
    # test_mesh_operations(nodes_number)
    # print("\n\n")
    test_mesh_shortcuts(interval_number)