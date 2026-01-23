import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_squared_error, 
    r2_score
)

def plot_regression(obs, pred, title='', fig_out_fpath=None, hide=False):

    # Calculate regression metrics
    r2 = r2_score(obs, pred)
    mse = mean_squared_error(obs, pred)
    rmse = np.sqrt(mse)
    bias = np.mean(pred - obs)

    # Set square figure
    fig, ax = plt.subplots(figsize=(5, 5))  # Make figure square
    ax.set(aspect='equal')

    # Set axis limits to be equal and tight
    min_val = min(obs.min(), pred.min())
    max_val = max(obs.max(), pred.max())
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    sns.scatterplot(ax=ax, x=obs, y=pred, color='black')
    ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
    ax.set_xlabel('Observed AGB (Mg/ha)', fontsize=11)
    ax.set_ylabel('Predicted AGB (Mg/ha)', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.text(.02, .98, 
            s = f'R²: {r2:.2f}\nRMSE: {rmse:.2f} Mg/ha\nBias: {bias:.2f} Mg/ha',
            ha='left', va='top', 
            transform=ax.transAxes,
            fontsize=11)

    fig.tight_layout()

    if fig_out_fpath:
        fig.savefig(fig_out_fpath, dpi=400, bbox_inches='tight', pad_inches=0.05)
    
    if hide:
        plt.close(fig)
    else:
        plt.show()