import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

def load_dataframe(path):
    """
    Load DataFrame from specific path
    """       
    if isinstance(path, str):
        return pd.read_parquet(path)


def event_time_conversion(dataframe, col, time_format, **kwargs):
    """
    Convert dataframe column to datetime and create columns: event_month, event_day and event_hour
    """    
    # Get keyword arguments
    column_action = kwargs.pop('column_action', 'remove')

    # Convert column to datetime
    dataframe[col]  = pd.to_datetime(dataframe[col], format=time_format)

    # Create event_month column
    dataframe['event_month'] = dataframe[col].dt.month   

    # Create event_day column
    dataframe['event_day'] = dataframe[col].dt.day    

    # Create event_hour column
    dataframe['event_hour'] = dataframe[col].dt.hour     

    # Based on colum_action keyword argument, remove/keep column to transform.
    if column_action == 'remove':
        dataframe.drop(columns=[col], inplace=True)

    return dataframe


def expand_category_code(dataframe, col, **kwargs):
    """
    Expand dataframe category code by level
    """    
    # Get keyword arguments
    column_action = kwargs.pop('column_action', 'remove')
    label_level = kwargs.pop('label_level', 'L')

    # Create DataFrame with expand categories. 
    categories = dataframe[col].str.split('.', expand=True)  

    # Get total number of categories
    n_categories = categories.shape[1]

    # List with level labels
    category_label = []                                             
    for k in range(n_categories):
        category_label.append(label_level + str(k))

    # Asign category label to categories dataframe column.
    categories.columns = category_label                    

    # Concatenate input dataframe with categories dataframe
    dataframe = pd.concat([dataframe, categories], axis=1)                      

    # Remove column "col" and return dataframe.
    if column_action == 'remove':
        dataframe.drop(columns=col, inplace=True) 

    return dataframe     


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
    if column_action == 'remove':
        dataframe = dataframe[dataframe[col] < outlier_threshold]
    
    return dataframe


def features_to_keep(dataframe, col, n_features):
    """
    This function keeps the head n_features features and assigns 'others' to the least represented.
    """    
    features = dataframe[col].value_counts(normalize=True)[:n_features].index.to_list()
    dataframe.loc[~(dataframe[col].isin(features)), col] = 'others'
    return dataframe


def plot_head_brands_event_type(dataframe, group_condition, normalize=False, **kwargs):
    """
    This function plot event_types of specified top brands along time (day/hour)
    """ 

    filename = kwargs.pop('filename', './fig/brands_view_cart_purchase_by_' + group_condition + '.png') 
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20,6))

    df_view = dataframe[dataframe['event_type'] == 'view'].copy()
    df_cart = dataframe[dataframe['event_type'] == 'cart'].copy()
    df_purchase = dataframe[dataframe['event_type'] == 'purchase'].copy()
    
    norm_value = 1
    # Normalize plots to view by percent.
    if normalize:
        norm_value = dataframe.groupby(group_condition).size()

    for k in (dataframe['brand'].unique()): 
        (df_view[df_view['brand'] == k].groupby(group_condition).size()/norm_value).plot(label=k, 
                                                                                         ax=axes[0], 
                                                                                         ylabel='#Views', 
                                                                                         grid='both', 
                                                                                         legend=True,
                                                                                         title='Top brand views per ' + group_condition)
                                                                                                
        (df_cart[df_cart['brand'] == k].groupby(group_condition).size()/norm_value).plot(label=k, 
                                                                                         ax=axes[1], 
                                                                                         ylabel='#Carts', 
                                                                                         grid='both', 
                                                                                         legend=True,
                                                                                         title='Top brand carts per ' + group_condition)

        (df_purchase[df_purchase['brand'] == k].groupby(group_condition).size()/norm_value).plot(label=k, 
                                                                                                 ax=axes[2], 
                                                                                                 ylabel='#Purchase', 
                                                                                                 grid='both', 
                                                                                                 legend=True,
                                                                                                 title='Top brand purchase per ' + group_condition)    
    fig.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_head_customers_event_type(dataframe, n_customers, customers_id, group_condition, **kwargs):
    """
    This function plot event_types of specified head customers along time (day/hour)
    """ 
    filename=kwargs.pop('filename', './results/customer_analysis/event_type_for_head_customers_by_' + group_condition +'.png')
    fig, axes = plt.subplots(nrows=1, ncols=n_customers, figsize=(20,6))

    for jj in range(n_customers):
        customer = dataframe[dataframe['user_id'] == customers_id[jj]]
        for k in (dataframe['event_type'].unique()):
            customer[customer["event_type"]==k].groupby(group_condition).size().plot(label=k, 
                                                                                    ax=axes[jj],  
                                                                                    grid='both', 
                                                                                    legend=True,
                                                                                    title='Event per ' + str(group_condition) + ' client id: ' + str(customers_id[jj]))
    fig.tight_layout()
    fig.savefig(filename)
    plt.close()


def plot_event_types(dataframe, event_types, group_condition, axes=None,  **kwargs):
    """
    Plot event_type considering group_condition.
    This function is useful to view event_types alogn time (day, hour)
    """    

    title_str = kwargs.pop('title_str', 'Percent of smartphones views, carts and purchase by day' )
    filename = kwargs.pop('filename', './fig/smartphones_views_carts_purchase_in_month.png')    

    max_by_day = dataframe.groupby(group_condition).size()
    max_by_day = 1
    fig = plt.figure()
    for k in range(len(event_types)):
        key = event_types[k]
        if axes == None:
            (dataframe[dataframe['event_type'] == key].groupby(group_condition).size()/max_by_day).plot(label=key, 
                                                                                                        grid='both')
        else:
            (dataframe[dataframe['event_type'] == key].groupby(group_condition).size()/max_by_day).plot(label=key, 
                                                                                                        grid='both',
                                                                                                        ax=axes)

    plt.legend()
    plt.ylabel(str(event_types))
    plt.title(title_str)
    fig.tight_layout()
    plt.savefig(filename)
    plt.close()    


def plot_head_values(dataframe, cols, n_values, **kwargs):
    """
    Plot head values of specific column of DataFrame.
    """     
    plot_type = kwargs.pop('plot_type', 'barh')
    title_str = kwargs.pop('title_str', str(cols) + '_top values')
    filename = kwargs.pop('filename', str(cols) + '_top_values.png')

    
    fig = plt.figure()
    dataframe[cols].value_counts(normalize=True).head(n_values).plot(kind=plot_type, 
                                                                 title=title_str, 
                                                                 grid='both', 
                                                                 xlim=[0, 1])
                                                                 
    fig.tight_layout()
    plt.savefig(filename)
    plt.close()  


def plot_column_frequency(dataframe, col, plot_type, **kwargs):
    """
    Plot column ocurrence (value counts) and save it to file. 
    """   
    plt.figure() 
    # Get keyword arguments    
    title_str = kwargs.pop('title_str', col + ' column normalized value counts')
    filename = kwargs.pop('filename', col + '.png')    
    tmp = dataframe[col].value_counts(normalize=True)
    plotted = tmp.sort_values().plot(kind = plot_type,
                                grid='both',
                                title=title_str).get_figure()
    plotted.tight_layout()
    plotted.savefig(filename)
    plt.close()


def plot_data_distribution(dataframe, col, plot_type, ax=None, **kwargs):
    """
    Plot data distribution and save to a file. 
    """      
    # Get keyword arguments    
    title_str = kwargs.pop('title_str', col + ' distribution')
    filename = kwargs.pop('filename', col + '_distribution.png')  
    
    stats = pd.DataFrame(dataframe[col].describe())

    sns.set_style("whitegrid")
    if ax == None:
        fig, ax = plt.subplots(nrows=1, ncols=1)

    if plot_type == 'violinplot':   
        plotted = sns.violinplot(x=dataframe[col], ax=ax)        
    elif plot_type == 'boxplot':
        plotted = sns.boxplot(x=dataframe[col], palette="Set3", ax=ax)

    inserted_table = plotted.table(cellText=np.round(stats.values,2),
                     colWidths = [0.2],
                     rowLabels=stats.index,
                     colLabels=[col],
                     cellLoc = 'center', rowLoc = 'right',
                     loc='lower right')

    inserted_table.set_fontsize(8)
    plotted.set_title(title_str)
    if ax == None:
        fig.tight_layout()
        fig.savefig(filename)                
    else:
        fig = plotted.get_figure()
        fig.tight_layout()
        fig.savefig(filename)

    plt.close()      


def set_project_folders():
    """
    Create project directories to save images file. 
    """     
    # Create results folder if not exist
    project_dir_path = os.path.dirname(os.path.realpath(__file__))

    results_dir = project_dir_path + '/results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Creat subdirs in results for save images.
    results_sub_dirs = [results_dir + '/general_analysis', results_dir + '/customer_analysis', results_dir + '/samsung_analysis']

    for dirname in results_sub_dirs:
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    return results_sub_dirs    
