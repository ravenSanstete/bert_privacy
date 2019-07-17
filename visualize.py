import matplotlib.pyplot as plt
import seaborn as sns; sns.set_style("white")

from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
import numpy as np




def visualize():
    S_0 = np.load("data/part_fake_2/hand.0.gpt.npy")[:1000]
    S_1 = np.load("data/part_fake_2/hand.0.gpt.npy")[:1000]
    T_0 = np.load("hand.0.ground.npy")[:1000, :]
    T_1 = np.load("hand.1.ground.npy")[:1000, :]
    X = np.concatenate([S_0, S_1, T_0, T_1], axis = 0)
    print(X.shape)
    pca = PCA(n_components=2)
    pca.fit(X)
    S_0, S_1 = pca.transform(S_0), pca.transform(S_1)
    T_0, T_1 = pca.transform(T_0), pca.transform(T_1)
    print(S_0.shape)
    print(T_0.shape)
    hls = sns.color_palette("hls", 8)
    fig, ax = plt.subplots(1, 1, figsize = (10, 10))
    for i, S in enumerate([T_0, T_1]):
        plt.scatter(S[:, 0], S[:, 1], hls[i])
    plt.savefig("visual/embedding.gpt.mds.png", dpi = 108)



if __name__ == '__main__':
    visualize()
