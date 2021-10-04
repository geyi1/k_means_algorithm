import numpy as np
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt


# distance between two points
def get_distance(a, b):
    distance = np.linalg.norm(a - b)
    return distance


def assign_centroid(point, centroids):
    res = 0
    distance = float('inf')
    # calculate distance to every centroid
    # pick the centroid with the smallest distance
    # there will be at least one centroid
    for position in range(len(centroids)):
        centroid = centroids[position]
        if distance > get_distance(point, centroid):
            distance = get_distance(point, centroid)
            res = position
    return res


def initial_centroids(data, k):
    # shuffle the position of points and pick the first k points as centroids
    centroids = data.copy()
    np.random.shuffle(centroids)
    return centroids[0:k]


def move_centroids(assignment, data, centroids):
    # centroids 
    # for each centroid, calculate the sum of all the dimensions
    # from every point that is assigned to the specific centroid
    for i in range(len(assignment)):
        centroids[assignment[i]] = [x + y for x, y in zip(centroids[assignment[i]], data[i])]

    # then divide the sum by number of points assigned to the specific centroid
    for j in range(len(centroids)):
        count = assignment.count(j)
        centroids[j] = [each / count for each in centroids[j]]


def get_distortion(assignment, data, centroids):
    # get the distortion, given the assignment and centroids
    distortion = 0
    for i in range(len(assignment)):
        distortion += pow(get_distance(data[i], centroids[assignment[i]]), 2)
    distortion /= len(assignment)
    return distortion


###########
# problem one part 1 solution
###########
def k_mean(data, k, assigned_centroids=None):
    if len(data) == 0:
        return None, None

    # for i in range(trials):
    if assigned_centroids is None:
        # take k random points in dataset as centroids
        centroids = initial_centroids(data, k)
    else:
        # use given centroids
        centroids = assigned_centroids

    prev_assignment = []
    while 1:
        assignment = []

        # centroid assignment to each of the other points
        for idx in range(len(data)):
            centroid = assign_centroid(data[idx], centroids)
            assignment.append(centroid)

        # recompute k centroids while fixing the cluster points
        reset = [0.0 for loop in range(data[0].size)]
        centroids = np.tile(reset, (k, 1))

        move_centroids(assignment, data, centroids)
        current_distortion = get_distortion(assignment, data, centroids)
        print(current_distortion)

        # when assignment does not change, stop and return the result
        if prev_assignment == assignment:
            break
        else:
            prev_assignment = assignment

    return centroids, assignment


if __name__ == '__main__':
    data = load_breast_cancer().data
    distortion = []
    cluster_count = []
    # for i in range(1, 8):
    for i in range(2, 8):
        centroids, assignment = k_mean(data, i)
        distortion.append(get_distortion(assignment, data, centroids))
        cluster_count.append(i)
    plt.scatter(cluster_count, distortion)
    plt.xlabel("# of k")
    plt.ylabel("distortion")
    plt.show()

    data = np.array([[1, 0], [2, 0], [3, 0], [4, 0]])
    distortion = []
    cluster_count = []
    for i in range(2, 3):
        centroids, assignment = k_mean(data, i, np.array([[2, 0], [4, 0]]))
        distortion.append(get_distortion(assignment, data, centroids))
        cluster_count.append(i)
    plt.scatter(data[:, 0], data[:, 1], c=assignment)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=150)
    plt.show()
