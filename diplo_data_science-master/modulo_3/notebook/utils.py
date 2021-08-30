import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)
import plotly.express as px

def treat_outiers(dataframe, col, **kwargs):
    """
    Treat outliers considering interquartile range.
    """     
    # Get keyword arguments
    column_action = kwargs.pop('column_action', 'remove')

    q1 = dataframe[col].quantile(0.25)
    q3 = dataframe[col].quantile(0.75)
    iqr = q3 - q1
    outlier_threshold = q3 + (iqr *1.5)
    outlier_indexes = 0
    if column_action == 'remove':
        outlier_indexes = np.where(dataframe[col] > outlier_threshold)[0]
        dataframe = dataframe[dataframe[col] < outlier_threshold]        
    
    return dataframe, outlier_indexes 
	

def quantile_target_distribution_relative(data, column, nq=5, target='TARGET'):
    X = data[[column, target]].copy()
    if X[column].nunique() > nq:
        try:
            X['INTERVAL'] = pd.qcut(X[column], np.linspace(0,1,nq+1)).map(str)
        except:
            X['INTERVAL'] = pd.cut(X[column], 
                                   np.unique(np.quantile(X[column],np.linspace(0, 1, nq+1), interpolation='linear')))
    else:
        X['INTERVAL'] = X[column]

    X = X.groupby('INTERVAL')[target].value_counts(normalize=True).rename('PROPORTION').reset_index()
    X[target] = X[target].astype(str)
    X = X.sort_values(['INTERVAL', target])
    return X

def quantile_target_distribution_plot_relative(data, column, nq=5, target='TARGET'):
    fig = px.bar(quantile_target_distribution_relative(data, column, nq, target), 
                 x='INTERVAL', y='PROPORTION', color=target, barmode='relative',
                height=400)
    fig.update_layout(title_text=f'{column}')
    fig.update_yaxes(range=[0,1])    
    return fig
	

def plot_DBSCAN(X1,labels_1,db_1,n_clusters_1, axes):
    # Armamos una mascara, con unos en los datos que son CORES.
    core_samples_mask_1 = np.zeros_like(db_1.labels_, dtype=bool)
    core_samples_mask_1[db_1.core_sample_indices_] = True
    # Plot result

    # Black removed and is used for noise instead.
    unique_labels = set(labels_1)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels_1 == k)

        xy = X1[class_member_mask & core_samples_mask_1]
        axes.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = X1[class_member_mask & ~core_samples_mask_1]
        axes.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)
    axes.set_title('#clusters: {:d} (eps: {:.1f}, min_samples: {:d})'.format(n_clusters_1, 
                                                                             db_1.eps,
                                                                             db_1.min_samples))
    
    return