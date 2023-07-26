import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

def plot_embeddings_pca(embeddings, labels):
    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(embeddings.squeeze().cpu().numpy())

    label_names = [
        "Fact",
        "Argument",
        "Precedent",
        "Ratio",
        "RulingL",
        "RulingP",
        "Statute",
    ]

    plt.figure(figsize=(10, 8))
    for i, label in enumerate(label_names):
        mask = labels.squeeze().cpu().numpy() == i
        plt.scatter(embeddings_pca[:, 0][mask], embeddings_pca[:, 1][mask], label=label)
    plt.legend()
    plt.title('PCA Plot')
    plt.show()


def plot_embeddings_tsne(embeddings, labels):
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_tsne = tsne.fit_transform(embeddings.squeeze().cpu().numpy())

    label_names = [
        "Fact",
        "Argument",
        "Precedent",
        "Ratio",
        "RulingL",
        "RulingP",
        "Statute",
    ]

    plt.figure(figsize=(10, 8))
    for i, label in enumerate(label_names):
        mask = labels.squeeze().cpu().numpy() == i
        plt.scatter(embeddings_tsne[:, 0][mask], embeddings_tsne[:, 1][mask], label=label)
    plt.legend()
    plt.title('t-SNE Plot')
    plt.show()


def plot_embeddings_umap(embeddings, labels):
    reducer = umap.UMAP(random_state=42)
    embeddings_umap = reducer.fit_transform(embeddings.squeeze().cpu().numpy())

    label_names = [
        "Fact",
        "Argument",
        "Precedent",
        "Ratio",
        "RulingL",
        "RulingP",
        "Statute",
    ]

    plt.figure(figsize=(10, 8))
    for i, label in enumerate(label_names):
        mask = labels.squeeze().cpu().numpy() == i
        plt.scatter(embeddings_umap[:, 0][mask], embeddings_umap[:, 1][mask], label=label)
    plt.legend()
    plt.title('UMAP Plot')
    plt.show()
