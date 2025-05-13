import seaborn as sns
import matplotlib.pyplot as plt

def plot_chi2_scores(chi2_df, title="Feature Importance (Chi² Scores)", palette="Set2"):
    """
    Plots Chi² scores for features using a horizontal bar plot.

    Parameters:
        chi2_df (pd.DataFrame): DataFrame with 'Feature' and 'Chi2 Score' columns.
        title (str): Title of the plot.
        palette (str): Seaborn color palette.
    """
    # Sort for better visualization
    chi2_df_sorted = chi2_df.sort_values(by="Chi2 Score", ascending=False)

    # Plot setup
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="whitesmoke")

    sns.barplot(
        x="Chi2 Score",
        y="Feature",
        data=chi2_df_sorted,
        hue='Feature',
        ax=ax,
        palette=palette
    )

    ax.set_title(title, fontsize=16, weight="bold")
    ax.set_xlabel("Chi² Score")
    ax.set_ylabel("Feature")
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)

    return fig
