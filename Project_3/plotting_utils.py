import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import seaborn as sns


def show_image(images,num_images,image_size=(64,64)) -> plt.Figure:

    if num_images==5:
        fig, axes = plt.subplots(1, 5, figsize=(10, 3))

        # Make sure axes is iterable
        if num_images == 1:
            axes = [axes]

        # Plot the images
        for i in range(5):
            ax = axes[i]
            ax.imshow(images[i].reshape(image_size[0],image_size[1]), cmap='gray')
            ax.set_title(f"Image {i+1}", fontsize=8)
            ax.axis('off')

        fig.tight_layout()
        return fig
    else:
        fig, axes = plt.subplots(2, 5, figsize=(10, 4))

        # Make sure axes is iterable
        if num_images == 1:
            axes = [axes]
        axes=axes.flatten()
        # Plot the images
        for i in range(10):
            ax = axes[i]
            ax.imshow(images[i].reshape(image_size[0],image_size[1]), cmap='gray')
            ax.set_title(f"Image {i+1}", fontsize=8)
            ax.axis('off')

        fig.tight_layout()
        return fig

def plot_explained_variance(pca, threshold=0.95):
    """
    Plots cumulative explained variance using seaborn with style and optimal component indication.

    Parameters:
    - pca: a fitted sklearn PCA object
    - threshold: float, variance threshold to mark optimal number of PCs

    Returns:
    - fig: matplotlib Figure object
    """
    sns.set(style='whitegrid')

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = len(cumulative_variance)

    # Find the optimal number of components to reach the threshold
    optimal_pc = np.argmax(cumulative_variance >= threshold) + 1

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the cumulative variance line
    sns.lineplot(x=np.arange(1, n_components + 1), y=cumulative_variance,
                 marker='o', ax=ax, linewidth=2, color="#4B8BBE")

    # Plot horizontal threshold line (e.g., 95%)
    ax.axhline(y=threshold, color='gray', linestyle='--', label=f'{int(threshold * 100)}% Variance')

    # Plot vertical line at optimal number of components
    ax.axvline(x=optimal_pc, color='red', linestyle='--', label=f'Optimal PC = {optimal_pc}')

    # Highlight the optimal point
    ax.plot(optimal_pc, cumulative_variance[optimal_pc - 1], 'ro', markersize=8)

    # Labels and title
    ax.set_xlabel('Number of Components', fontsize=12)
    ax.set_ylabel('Cumulative Explained Variance', fontsize=12)
    ax.set_title('Explained Variance vs Number of Principal Components', fontsize=14, fontweight='bold')

    # Limits and ticks
    ax.set_xlim(1, n_components)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right')
    ax.grid(True)

    return fig,optimal_pc


def show_eigenfaces(pca, image_shape, n_components=5):
    if n_components==5:
        fig, axes = plt.subplots(1, 5, figsize=(10, 3))
        for i in range(5):
            ax = axes[i]
            ax.imshow(pca.components_[i].reshape(image_shape), cmap='gray')
            ax.set_title(f"PCA {i+1}", fontsize=8)
            ax.axis('off')

    else:
        fig, axes = plt.subplots(2, 5, figsize=(10, 4))
        axes=axes.flatten()
        for i in range(10):
            ax = axes[i]
            ax.imshow(pca.components_[i].reshape(image_shape), cmap='gray')
            ax.set_title(f"PCA {i+1}", fontsize=8)
            ax.axis('off')
    fig.tight_layout()
    return fig

def show_image_comparison(original, reconstructed, image_shape):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(original.reshape(image_shape), cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(reconstructed.reshape(image_shape), cmap='gray')
    axes[1].set_title("Reconstructed Image")
    axes[1].axis('off')

    st.subheader("Image Reconstruction")
    st.pyplot(fig)

def show_reconstructed_image(img_resized,img_reconstructed,target_shape):
  fig, axes = plt.subplots(1, 2, figsize=(8, 4))
  axes[0].imshow(img_resized, cmap='gray')
  axes[0].set_title("Original (Resized)")
  axes[0].axis('off')

  axes[1].imshow(img_reconstructed.reshape(target_shape), cmap='gray')
  axes[1].set_title("Reconstructed")
  axes[1].axis('off')
  return fig
