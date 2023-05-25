import sys
import mykmeanssp as dm
import numpy as np

# gets array and print : X_1, X_2 ...
def printarr(arr):
    n = len(arr)
    for i in range (n):
        print(arr[i], end="")
        if i != n-1:
            print(",", end="")
    print("")

# matrix printer
def printmatx(mat):
    n = len(mat)
    m = len(mat[0])
    for i in range (n):
        for j in range(m):
            print("%.4f" % mat[i][j], end="")
            if j != m-1:
                print(",", end="")
        print("\n")

        
# gets 2 vector in R^d
# returns the euclidean distance of them
def euclidean_dist(arr_1, arr_2):
    dist = 0
    for i in range(len(arr_1)):
        dist += (float(arr_1[i]) - float(arr_2[i])) ** 2
    return dist ** 0.5


# gets data file of vectors
# return list with all the vectors as arrays, and k = len of vectors, size = number of vectors
def data_builder(filepath):
    f = open(filepath, "r")
    data_vectors = []
    line = f.readline()
    k = len(line.split("\n")[0].split(','))
    size = 0
    while line != "":
        size += 1
        arr_line = line.split("\n")[0].split(',')
        data_vectors.append(arr_line)
        line = f.readline()
    return data_vectors, k, size

# gets data file of vectors
# return list with all the vectors as float arrays
def data_builder_floats(filepath):
    f = open(filepath, "r")
    data_vectors = []
    line = f.readline()
    size = 0
    k = len(line.split("\n")[0].split(','))
    while line != "":
        size += 1
        arr_line = line.split("\n")[0].split(',')
        arr_line = [float(x) for x in arr_line]
        data_vectors.append(arr_line)
        line = f.readline()
    return data_vectors, k, size

def find_max_gap(arr):
    max_gap = 0
    max_gap_ix = 0
    curr_gap = 0
    for i in range(1,1+len(arr)//2):
        curr_gap = arr[i]-arr[i-1]
        if curr_gap > max_gap:
            max_gap = curr_gap
            max_gap_ix = i 
    return i

def write_to_file(data_points):
    f = open("file_1.txt", 'w')
    for row in data_points:
        row_str = ','.join('{:.4f}'.format(element) for element in row)
        f.write(row_str+"\n")

def first_vec(data):
    return np.random.choice(range(0, len(data)))

def prob(data, indx_list):
    l = 0
    probs = [0 for i in range(len(data))]
    for v in data:
        min_val = np.inf
        for i in indx_list:
            curr_val = euclidean_dist(v, data[i])
            if curr_val < min_val:
                min_val = curr_val
        probs[l] = min_val
        l += 1
    n = sum(probs)
    return [probs[i]/n for i in range(len(probs))]

def kmeans_pp_cen_init(k, data_vec):
    np.random.seed(0)
    indx = [i for i in range(len(data_vec))]
    k_vals = []
    k_vals.append(first_vec(data_vec))
    for i in range(k-1):
        probs = prob(data_vec, k_vals)
        k_vals.append(np.random.choice(indx, p=probs))
    return k_vals
    
# recieve input from user
if len(sys.argv) == 4:
    input_file = sys.argv[3]
    k = int(sys.argv[1])
    goal = sys.argv[2]
else:
    goal = sys.argv[1]
    input_file = sys.argv[2]
    k = 0

# make float matrix from data
data_points, d, N= data_builder_floats(input_file)

if goal == "spk":
    mat = dm.gl(data_points)
    mat = dm.jacobi(mat, 0, 0)
    arr = mat.pop(0)
    if k == 0:
        K = find_max_gap(arr)
    else:
        K = k
    # using first K vectors
    mat = np.array(mat)
    sorted_indices = np.argsort(mat[0])
    sorted_matrix = mat[:, sorted_indices]
    d_mat = mat[:, :K].tolist()
    arr_idx = kmeans_pp_cen_init(K, d_mat)
    mean_arr_str = ",".join([str(x) for x in arr_idx])
    print(mean_arr_str)
    mat = dm.spk(d_mat, arr_idx, N, K)

if goal == "wam":
    mat = dm.wam(data_points)

if goal == "ddg":
    mat = dm.ddg(data_points)

if goal =="gl":
    mat = dm.gl(data_points)

if goal =="jacobi":
    mat = dm.jacobi(data_points, 0, 0)
    
printmatx(mat)
