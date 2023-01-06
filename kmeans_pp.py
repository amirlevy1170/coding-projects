import sys
# from hw_1py import kmeans
import numpy as np
import pandas as pd
import kmeans as dm

# gets 2 vector in R^d
# returns the euclidean distance of them
def euclidean_dist(arr_1, arr_2):
    dist = 0
    for i in range(len(arr_1)):
        dist += (float(arr_1[i]) - float(arr_2[i])) ** 2
    return dist ** 0.5


# gets data file of vectors
# return list with all the vectors as arrays
# first list is K vectors as clusters
def data_builder(filepath):
    f = open(filepath, "r")
    data_vectors = []
    line = f.readline()
    k = len(line.split("\n")[0].split(','))
    while line != "":
        arr_line = line.split("\n")[0].split(',')
        data_vectors.append(arr_line)
        line = f.readline()
    return data_vectors, k


# gets 2 data files of vectors
# creates a new txt file : "file_1.txt" that contains the inner join vectors of both given files
# delete cord 0 - represent the key of the inner
def build(file_1, file_2):
    filepath_1 = file_1
    filepath_2 = file_2

    data_1, k_1 = data_builder(filepath_1)
    data_2, k_2 = data_builder(filepath_2)
    data_1 = np.array(data_1)
    data_2 = np.array(data_2)
    data_file_1 = pd.DataFrame()
    data_file_2 = pd.DataFrame()

    # making the input files to pandas Data frame
    for j in range(k_1):
        arr = []
        for i in range(len(data_1)):
            arr.append(data_1[i][j])
        label = "cord_" + str(j)
        data_file_1.insert(j, label, arr)
    for j in range(k_1):
        arr = []
        for i in range(len(data_1)):
            arr.append(data_2[i][j])
        label = "cord_" + str(j)
        data_file_2.insert(j, label, arr)

    # merging two data file using inner join
    inner_join_data = pd.merge(data_file_1, data_file_2, on="cord_0")

    inner_join_data = inner_join_data.set_index("cord_0")
    inner_join_data = inner_join_data.sort_values("cord_0")

    # write to new input_file.txt
    f = open("file_1.txt", 'w')
    res = inner_join_data.to_string(header=False, index=False, index_names=False).split('\n')
    # Adding a comma in between each value of list
    res = [','.join(ele.split()) for ele in res]
    vectors = ""
    for i in range(len(res)):
        vectors += res[i]
        vectors += "\n"
    f.write(vectors)


# gets k - number of centroids
# return the indexes of the first K centroids using the kmeans++ implementation
def kmeans_pp_cen_init(k, filepath="file_1.txt"):
    data_vec, cord_length = data_builder(filepath)
    vec_arr_chosen = []
    # reset the random seed
    np.random.seed(0)
    sample = [i for i in range(len(data_vec))]
    # choosing uniformly the first center
    vec_arr_chosen.append(np.random.choice(a=sample, size=1)[0])
    j = 1
    while j < k:
        prob_arr = []
        sum = 0
        for vec in data_vec:
            ind = data_vec.index(vec)
            # make sure that no center is being chosen twice
            if ind in vec_arr_chosen:
                prob_arr.append(0)
            else:
                # calculate D(x) and append it to the probability array
                min_dist = 4 ** 10
                for index in vec_arr_chosen:
                    d = euclidean_dist(vec, data_vec[index])
                    if d < min_dist:
                        min_dist = d
                sum += min_dist
                prob_arr.append(min_dist)
        prob_arr = np.array(prob_arr)
        # make the probability array to valid probability function
        prob_arr = prob_arr / sum
        # choosing new center using the prob' func'
        vec_arr_chosen.append(np.random.choice(a=sample, size=1, p=prob_arr)[0])
        j += 1
    return vec_arr_chosen, cord_length



# recieve input from user
if len(sys.argv) == 6:
    input_file_1 = sys.argv[4]
    input_file_2 = sys.argv[5]
    k = int(sys.argv[1])
    iter = int(sys.argv[2])
    eps = sys.argv[3]
else:
    input_file_1 = sys.argv[3]
    input_file_2 = sys.argv[4]
    k = int(sys.argv[1])
    eps = sys.argv[2]
    iter = 300

# make the new data file using inner join
build(input_file_1, input_file_2)

# find the new K centroids using k-means ++ algorithem
mean_arr, cord_len = kmeans_pp_cen_init(k)
mean_arr_str = ",".join([str(x) for x in mean_arr])

# calculate the k centroids using C module
cKmean = dm.fit(k, iter, 0, mean_arr_str, cord_len)

# printing the indexes of the randomly choosen first k means
m = 0
print(mean_arr_str)

# printing the final centroids
for i in range (k):
    print("%.4f" % float(cKmean[m]), end="")
    m += 1
    for j in range(cord_len-1):
        print(",%.4f" % float(cKmean[m]), end="")
        m += 1
    print("\n")
