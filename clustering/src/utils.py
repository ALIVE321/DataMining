import pandas as pd
import numpy as np
import configs as cfg


def read_data():
    df = pd.read_csv(cfg.data_path, sep=',')
    data = np.array(df[[f"DIM_{i}" for i in range(cfg.dims)]])
    return data


def distance(v1, v2):
    d = 0
    for i in range(len(v1)):
        d += np.square(v1[i] - v2[i])
    return np.sqrt(d)


def init_centroids(data):
    n, _ = data.shape
    centroids = np.random.choice(n, cfg.K, replace=False)
    return np.array(data[centroids])


def nearest_centroid(v, centroids):
    min_dist = 0x7fffffff
    res = -1
    for i in range(cfg.K):
        d = distance(v, centroids[i])
        if min_dist >= d:
            min_dist = d
            res = i
    return res


def create_clusters(centroids, data):
    clusters = [[] for _ in range(cfg.K)]
    for i in range(len(data)):
        clusters[nearest_centroid(data[i], centroids)].append(i)
    return clusters


def new_centroids(clusters, data):
    return np.array([np.mean(data[clusters[i]], axis=0) for i in range(cfg.K)])


def get_cluster_res(centroids, data):
    res = []
    for i in range(len(data)):
        res.append(nearest_centroid(data[i], centroids))
    return res


def rearrange(centroids, data):
    clusters = create_clusters(centroids, data)
    radius = [np.mean([distance(data[j], centroids[i]) for j in clusters[i]]) for i in range(cfg.K)]
    print(radius)
    sorted_idx = sorted(list(range(cfg.K)), key=lambda x: radius[x])
    print(sorted_idx)
    ans = np.zeros(shape=(len(data),), dtype="int")
    for i in range(cfg.K):
        for j in clusters[i]:
            ans[j] = (sorted_idx[i])
    return ans


def cluster(data):
    centroids = init_centroids(data)
    diff = np.zeros(shape=(cfg.K,))
    diff[0] = 1
    iters = 0
    while diff.any():
        # if not iters % 10:
        print(f"Iteration {iters}")
        iters += 1
        clusters = create_clusters(centroids, data)
        prev_centroids = centroids
        centroids = new_centroids(clusters, data)
        diff = centroids - prev_centroids
    ans = rearrange(centroids, data)
    return ans


def write_res(res):
    output = {"id": list(range(len(res))), "category": res}
    df = pd.DataFrame(output)
    df.to_csv(cfg.out_path, sep=',', index=False)


if __name__ == "__main__":
    pass
