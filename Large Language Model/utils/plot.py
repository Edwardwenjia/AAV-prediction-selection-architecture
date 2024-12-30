
import os
import matplotlib.pyplot  as plt
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import seaborn as sns
import numpy as np
from utils.utils_f4f import heatmap, si_format
import matplotlib as mpl


def plot_results(preds_df, save_dir, plot_file_name, fig_size = (1.4,1.3), scatter_size = 1, line_color = 'red', alpha = 0.6):

    preds_df  =  preds_df.replace([np.inf, -np.inf], np.nan)
    preds_df  =  preds_df.fillna(0) 
    plt.figure(figsize = fig_size)

    x  =  preds_df['y_pred']  
    y  =  preds_df['y_true']   
    preds_df['y_pred'][:1000].corr(preds_df['y_true'][:1000])
    remove  =  np.isinf(x) & np.isinf(y)
    x  =  x[~remove]
    y  =  y[~remove]
    keep  =  (~np.isinf(x)) & (~np.isinf(y))
    x_both  =  x[keep]
    y_both  =  y[keep]
    y_missing  =  y[np.isinf(x)]
    x_missing  =  x[np.isinf(y)]
    kernel  =  gaussian_kde(np.vstack([
        x_both.sample(n = int(x_both.shape[0]*0.5), random_state = 1), 
        y_both.sample(n = int(y_both.shape[0]*0.5), random_state = 1)
    ]))
    c_both  =  kernel(np.vstack([x_both.values, y_both.values]))

    fig  =  plt.figure(figsize = (1.4,1.3), dpi = 150)
    gs  =  fig.add_gridspec(2, 2, left = 0.275, right = 0.95, bottom = 0.22, top = 0.95, 
                        width_ratios = [1, 6], height_ratios = [6, 1], hspace = 0., wspace = 0)
    ax  =  fig.add_subplot(gs[0, 1])
    ax.scatter(x_both, y_both, c = c_both, cmap = mpl.cm.inferno, s = 0.2, edgecolor = 'none', rasterized = True)
    ax.set_xticks([]); ax.set_yticks([])
    xlim  =  [-11, 12]
    bins  =  np.linspace(*xlim, 25)
    ax.set_xlim(xlim); ax.set_ylim(xlim)
    ax.text(0.03, 0.97, r'$r$  =  {:.4f}'.format(np.corrcoef(x_both, y_both)[0, 1]),
        transform = ax.transAxes, ha = 'left', va = 'top', fontsize = 7)

    ax.text(0.97, 0.01, 'n = {}'.format(si_format(len(x_both)), precision = 2, format_str = '{value}{prefix}',), 
            transform = ax.transAxes, ha = 'right', va = 'bottom', fontsize = 7)
    ax  =  fig.add_subplot(gs[0, 0])
    ax.hist(y_missing, bins = bins, edgecolor = 'none', orientation = 'horizontal', density = True, color = 'r')
    ax.set_ylim(xlim)
    ax.set_xticks([]); ax.set_yticks([-5, 0, 5, 10, 15,20])
    ax.text(0.97, 0.97, 'n = {}'.format(si_format(len(y_missing)), precision = 2, format_str = '{value}{prefix}',), 
            transform = ax.transAxes, ha = 'right', va = 'top', fontsize = 7, rotation = 90, color = 'r')
    ax.set_ylabel('Prediction', labelpad = 2,fontsize = 8)
    ax.tick_params(axis = 'both', labelsize = 7, length = 2, pad = 1)
    ax  =  fig.add_subplot(gs[1, 1])
    ax.hist(x_missing, bins = bins, edgecolor = 'none', density = True, color = 'r')
    ax.set_xlim(xlim)
    ax.set_xticks([-5, 0, 5, 10, 15,20]); ax.set_yticks([])
    ax.text(0.97, 0.8, 'n = {}'.format(si_format(len(x_missing)), precision = 2, format_str = '{value}{prefix}',), 
            transform = ax.transAxes, ha = 'right', va = 'top', fontsize = 7, color = 'r')
    ax.set_xlabel('Truth', labelpad = 2,fontsize = 8)
    ax.tick_params(axis = 'both', labelsize = 7, length = 2, pad = 1)
    ax  =  fig.add_subplot(gs[1, 0])
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.8, 0.8, 'Missing', transform = ax.transAxes, color = 'r', ha = 'right', va = 'top', fontsize = 7, clip_on = False)


    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{plot_file_name}.pdf'), dpi = 300, bbox_inches = 'tight')
    
    plt.show()







def fit_gaussian_mixture(preds_df, n_components, save_dir, gm_plot_file_name):
    """使用Gaussian Mixture拟合数据并绘制分布图"""
    data  =  np.array(preds_df['y_pred']) 
    data  =  data.reshape(-1, 1)  
    gmm  =  GaussianMixture(n_components = n_components)
    gmm.fit(data)
    means  =  gmm.means_    
    covariances  =  gmm.covariances_ # Get the mean and covariance matrix for each component
    x  =  np.linspace(-7, 8, data.shape[0])
    x  =  x.reshape(-1, 1)
    y  =  np.exp(gmm.score_samples(x))  # Generate the fitted distribution
    fig,ax  =  plt.subplots(figsize = (1.8,1.5),dpi = 100)
    plt.hist(data, bins = 100, density = True, alpha = 0.6, color  =  '#BAB3A3',label = 'Prediction') # Plot the raw data and the fitted distribution
    for i in range(n_components):   # Plot the normal distribution for each component
        component  =  np.exp(-(x - means[i]) ** 2 / (2 * covariances[i]))
        component /=  np.sqrt(2 * np.pi * covariances[i])
        component *=  gmm.weights_[i]
        if i  ==  0:
            ax.plot(x, component, '-', label = f'Distribution {i + 1}', color = '#F66E68', alpha = 1)
            ax.fill_between(x.flatten(), 0, component.flatten(), color = '#F66E68', alpha = 0.4)
        else:
            ax.plot(x, component, '-', label = f'Distribution {i + 1}', color = '#457B9D', alpha = 1)
            ax.fill_between(x.flatten(), 0, component.flatten(), color = '#457B9D', alpha = 0.4)

    plt.title('Prediction',fontsize = 10)
    ax.tick_params(axis = 'both', which = 'both', length = 2,labelsize = 10)
    ax.set_xticks([-5, 0, 5, ]); 
    plt.ylabel('Density',labelpad = 1,fontsize = 10)
    plt.xlabel('Nor_package',labelpad = 1,fontsize = 10)
    plt.legend(fontsize = 7,frameon = False)

    # Save the image
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, f'{gm_plot_file_name}.pdf'), dpi = 300, bbox_inches = 'tight')
    plt.close()
