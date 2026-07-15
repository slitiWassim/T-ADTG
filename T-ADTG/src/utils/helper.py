import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D





def save_tsne_plots(features, labels, epoch,
                    save_dir="log/ablation/",phase='test'):
    save_dir = save_dir + phase
    os.makedirs(save_dir, exist_ok=True)

    features = np.asarray(features)
    labels = np.asarray(labels)

    # -------------------------
    # 2D t-SNE
    # -------------------------
    tsne_2d = TSNE(
        n_components=2,
        perplexity=30,
        random_state=42,
        init="pca"
    )

    emb_2d = tsne_2d.fit_transform(features)

    plt.figure(figsize=(8, 6))

    scatter = plt.scatter(
        emb_2d[:, 0],
        emb_2d[:, 1],
        c=labels,
        cmap="coolwarm",
        alpha=0.7,
        s=10
    )

    plt.colorbar(scatter)
    plt.title(f"2D t-SNE Epoch {epoch}")

    plt.savefig(
        os.path.join(save_dir, f"epoch_{epoch:03d}_tsne2d.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # -------------------------
    # 3D t-SNE
    # -------------------------
    tsne_3d = TSNE(
        n_components=3,
        perplexity=30,
        random_state=42,
        init="pca"
    )

    emb_3d = tsne_3d.fit_transform(features)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        emb_3d[:, 0],
        emb_3d[:, 1],
        emb_3d[:, 2],
        c=labels,
        cmap="coolwarm",
        alpha=0.7,
        s=10
    )

    fig.colorbar(sc)

    ax.set_title(f"3D t-SNE Epoch {epoch}")

    plt.savefig(
        os.path.join(save_dir, f"epoch_{epoch:03d}_tsne3d.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()