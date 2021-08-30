import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder

from utils import (event_time_conversion,
                   expand_category_code,
                   plot_column_frequency,
                   plot_data_distribution,
                   plot_head_values,
                   treat_outiers,
                   plot_event_types,
                   features_to_keep,
                   plot_head_brands_event_type,
                   plot_head_customers_event_type)


HEAD_CATEGORIES_NUMBER = 10
HEAD_CUSTOMERS_NUMBER = 3  
HEAD_BRANDS_NUMBER = 5 

def general_analysis(input_dataframe, output_dirname):
    """
    This function perform general analysis over dataset and saves the results into output folder.
    """      
    df = input_dataframe

    # "event_time" column conversion
    df = event_time_conversion(dataframe=df, 
                               col='event_time', 
                               time_format='%Y-%m-%d %H:%M:%S UTC',
                               column_action='remove')

    # Focus on price distribution of category code nulls
    plot_data_distribution(dataframe= df[df['category_code'].isnull()],
                           col= 'price',
                           plot_type= 'violinplot',
                           title_str= 'Price of category_code Nulls',
                           filename= output_dirname + '/price_distribution_of_category_code_nulls.png')


    # Expand category code by level.
    df = expand_category_code(dataframe=df, col='category_code', column_action='remove')

    # Drop Duplicates.
    df.drop_duplicates(inplace=True) 

    # Electronics First level value counts
    plot_column_frequency(dataframe=df[df['L0'] == 'electronics'], 
                           col='L1', 
                           plot_type='barh',
                           title_str='Category code level frequency (L1)',
                           filename= output_dirname + '/category_code_first_level.png')

    # Create views/carts/purchase DataFrame.
    df_views = df[df['event_type'] == 'view'].copy()
    df_carts = df[df['event_type'] == 'cart'].copy()
    df_purchase = df[df['event_type'] == 'purchase'].copy()

    # Plot top values of views, carts and puchase
    plot_head_values(dataframe = df_views, 
                    cols=['L0', 'L1'], 
                    n_values=HEAD_CATEGORIES_NUMBER,
                    title_str='Views by category',
                    filename=output_dirname + '/views_by_category.png')

    plot_head_values(dataframe = df_carts, 
                    cols=['L0', 'L1'], 
                    n_values=HEAD_CATEGORIES_NUMBER,
                    title_str='Carts by category',
                    filename=output_dirname + '/carts_by_category.png')


    plot_head_values(dataframe = df_purchase, 
                    cols=['L0', 'L1'], 
                    n_values=HEAD_CATEGORIES_NUMBER,
                    title_str='Purchase by category',
                    filename=output_dirname + '/purchase_by_category.png')                                        

    # Create smartphone DataFrame.
    df_smartphones = df[df['L1'] == 'smartphone'].copy()

    # Plot price distribution of smarphones price.
    plot_data_distribution( dataframe= df_smartphones,
                            col= 'price',
                            plot_type= 'violinplot',
                            title_str= 'Smartphone price distribution',
                            filename= output_dirname + '/smartphones_price_distribution.png')  

    # Remove outliers prices of smartphone DataFrame.
    df_smartphones = treat_outiers(dataframe=df_smartphones, 
                                   col='price',
                                   column_action='remove')

    # Plot price distribution of smarphones price without outliers.
    plot_data_distribution( dataframe= df_smartphones,
                            col= 'price',
                            plot_type= 'violinplot',
                            title_str= 'Smartphone price distribution',
                            filename= output_dirname + '/smartphones_price_distribution_without_outliers.png')                  

    # Plot smartphones views/carts and purchase per day
    plot_event_types(dataframe=df_smartphones, 
                     event_types=['view', 'cart', 'purchase'], 
                     group_condition='event_day',
                     filename = output_dirname + '/smartphones_views_carts_purchase_in_month.png')

    # We keep the first HEAD_BRANDS_NUMBER and asing 'others' to the least represented
    df_smartphones = features_to_keep(dataframe=df_smartphones, col='brand', n_features=HEAD_BRANDS_NUMBER)

    # Plot head brands event_type per day
    plot_head_brands_event_type(dataframe=df_smartphones, 
                            group_condition='event_day', 
                            normalize=False, 
                            filename=output_dirname + '/brands_view_cart_purchase_by_event_day')

    # Plot head brands event_type per day
    plot_head_brands_event_type(dataframe=df_smartphones, 
                                group_condition='event_hour', 
                                normalize=False,
                                filename=output_dirname + '/brands_view_cart_purchase_by_event_hour') 

    return df

def retargeting_analysis(input_dataframe, output_dirname):
    """
    This function perform retargeting analysis for HEAD_CUSTOMERS_NUMBER.
    """          
    
    df = input_dataframe
    df_purchase = df[df['event_type'] == 'purchase'].copy()
    df_smartphones = df_purchase[df_purchase['L1'] == 'smartphone'].copy()

    print('Customer head analysis \r')
    heads_customer_analysis(dataframe=df, dirname=output_dirname[0])
    
    print('Samsung brand analysis \r')
    samsung_brand_analysis(dataframe=df_purchase, dirname=output_dirname[1])
    
def heads_customer_analysis(dataframe, dirname):
    """
    This function perform customer analysis for HEAD_CUSTOMERS_NUMBER.
    """     
    
    df = dataframe
    df_purchase = df[df['event_type'] == 'purchase'].copy()
    labelencoder = LabelEncoder()

    # View/cart/purchase profile of head customers.
    plot_head_customers_event_type(dataframe=df, 
                                   n_customers=HEAD_CUSTOMERS_NUMBER, 
                                   customers_id=df_purchase['user_id'].value_counts()[:HEAD_CUSTOMERS_NUMBER].index, 
                                   group_condition='event_day',
                                   filename=dirname + '/event_type_for_head_customers_by_event_day.png')

    plot_head_customers_event_type(dataframe=df, 
                                   n_customers=HEAD_CUSTOMERS_NUMBER, 
                                   customers_id=df_purchase['user_id'].value_counts()[:HEAD_CUSTOMERS_NUMBER].index, 
                                   group_condition='event_hour',
                                   filename=dirname + '/event_type_for_head_customers_by_event_hour.png')


    # Price profile of head customers
    customer_id = df_purchase['user_id'].value_counts()[:HEAD_CUSTOMERS_NUMBER].index


    fig0, axes0 = plt.subplots(nrows=1, ncols=HEAD_CUSTOMERS_NUMBER, figsize=(15,10))
    fig1, axes1 = plt.subplots(nrows=1, ncols=HEAD_CUSTOMERS_NUMBER, figsize=(20,6))
    fig2, axes2 = plt.subplots(nrows=HEAD_CUSTOMERS_NUMBER, ncols=1, figsize=(20,10))
    fig3, axes3 = plt.subplots(nrows=1, ncols=HEAD_CUSTOMERS_NUMBER, figsize=(20,6))

    for jj in range(HEAD_CUSTOMERS_NUMBER):
        customer = df[df['user_id'] == customer_id[jj]]

        f0 = sns.violinplot(x="price",data=customer,ax=axes0[jj], palette="Set3")
 
        customer_purchase = customer[customer['event_type'] == 'purchase']  
        f1 = sns.violinplot(x="brand",y="price",data=customer_purchase,ax=axes1[jj], palette="Set3", order=["samsung", "apple", "lg", "artel"])

        f2 = sns.violinplot(x="event_day", y="price", data=customer_purchase, ax=axes2[jj])
        axes2[jj].set_title('Customer id ' + str(customer_id[jj]))
                
        customer_purchase_ = customer[customer['event_type'] == 'purchase'].copy()                 
        customer_purchase_['brand_N'] = labelencoder.fit_transform(customer_purchase_['brand'])        
        customer_purchase_['user_session_N'] = labelencoder.fit_transform(customer_purchase_['user_session']) 
        customer_purchase_.drop(columns=['event_type', 'event_month', 'brand', 'user_id', 'user_session'], inplace=True)
        f3 = sns.heatmap(customer_purchase_.corr(), annot=True, ax=axes3[jj])
        axes3[jj].set_title('Corr. Customer id ' + str(customer_id[jj]))



    f0.get_figure().savefig(dirname + '/price_profile_head_customers.png')
    f1.get_figure().savefig(dirname + '/purchase_brand_by_customer.png')
    tmp = f2.get_figure()
    tmp.tight_layout()
    tmp.savefig(dirname + '/purchase_price_distribution_by_day_for_each_customer_.png')
    f3.get_figure().savefig(dirname + '/corrs_matrix_each_customer.png')
    plt.close()        

def samsung_brand_analysis(dataframe, dirname):
    """
    This function perform samsung brand analysis.
    """     

    df_purchase = dataframe

    # Plot price distribution of purchase in all data frame
    plot_data_distribution( dataframe= df_purchase,
                            col= 'price',
                            plot_type= 'violinplot',
                            title_str= 'Purchase price distribution',
                            filename= dirname + '/purchase_price_distribution.png')


    # We focus in purchase Samsung brand
    df_purchase_samsung = df_purchase[df_purchase['brand'] == 'samsung']

    # Analize Samsung price distribution
    plot_data_distribution( dataframe= df_purchase_samsung,
                            col= 'price',
                            plot_type= 'violinplot',
                            title_str= 'Samsung price distribution',
                            filename= dirname + '/samsung_price_distribution.png')

    
    # Group Samsung by price
    plt.figure()
    df_purchase_samsung.groupby('price').size().plot(grid='both',
                                                     title='Puchases samsung grouped by price')
    plt.tight_layout()
    plt.savefig(dirname + '/samsung_grouped_by_price.png')
    plt.close()   

    # Treat outliers of Samsung price
    df_purchase_samsung_without_outliers = treat_outiers(dataframe=df_purchase_samsung, 
                                                         col='price',
                                                         column_action='remove')


    # Verify Samsung price grouping after outlier treatment
    plt.figure()
    df_purchase_samsung_without_outliers.groupby('price').size().plot(grid='both',
                                                                      title='Puchases samsung grouped by price')
    plt.tight_layout()
    plt.savefig(dirname + '/samsung_grouped_by_price_without_outliers.png')
    plt.close()                                         

    # Samsung purchase per day
    plot_event_types(dataframe=df_purchase_samsung_without_outliers, 
                     event_types=['purchase'], 
                     group_condition='event_day',
                     title_str='Samsung purchases per day',
                     filename=dirname + '/samsung_per_day.png')      

    # Samsung purchase per hour
    plot_event_types(dataframe=df_purchase_samsung_without_outliers, 
                     event_types=['purchase'], 
                     group_condition='event_hour',
                     title_str='Samsung purchases per hour',
                     filename=dirname + '/samsung_per_hour.png')                                      


    # Analize which Samsung category level (L0, L1, L2) it's outside of interquartile range
    IQR = df_purchase_samsung['price'].quantile(0.75)- df_purchase_samsung['price'].quantile(0.25)

    plot_column_frequency( dataframe=df_purchase_samsung[df_purchase_samsung['price'] > 3*IQR], 
                            col='L0', 
                            plot_type='barh',
                            title_str='Samsung L0 ocurrence',
                            filename=dirname + '/samsung_L0_outliers.png')

    plot_column_frequency( dataframe=df_purchase_samsung[df_purchase_samsung['price'] > 3*IQR], 
                            col='L1', 
                            plot_type='barh',
                            title_str='Samsung L1 ocurrence',
                            filename=dirname + '/samsung_L1_outliers.png')                            

    plot_column_frequency( dataframe=df_purchase_samsung[df_purchase_samsung['price'] > 3*IQR], 
                            col='L2', 
                            plot_type='barh',
                            title_str='Samsung L2 ocurrence',
                            filename=dirname + '/samsung_L2_outliers.png')        

    # Analize which Samsung category level (L0, L1, L2) it's inside of interquartile range 
    plot_column_frequency( dataframe=df_purchase_samsung_without_outliers, 
                            col='L0', 
                            plot_type='barh',
                            title_str='Samsung L0 ocurrence',
                            filename=dirname + '/samsung_L0.png')

    plot_column_frequency( dataframe=df_purchase_samsung_without_outliers, 
                            col='L1', 
                            plot_type='barh',
                            title_str='Samsung L1 ocurrence',
                            filename=dirname + '/samsung_L1.png')                            

    plot_column_frequency( dataframe=df_purchase_samsung_without_outliers, 
                            col='L2', 
                            plot_type='barh',
                            title_str='Samsung L2 ocurrence',
                            filename=dirname + '/samsung_L2.png')   

    # We notice that "smartphone" it's the main caterory purchased of Samsung.
    # We focus on price distribution of Samsung smartphones.

    df_smartphones_samsung = df_purchase_samsung[df_purchase_samsung['L1'] == 'smartphone'].copy()
    
    plot_data_distribution( dataframe= df_smartphones_samsung[df_smartphones_samsung['brand'] == 'samsung'],
                            col= 'price',
                            plot_type= 'violinplot',
                            title_str= 'Samsung smartphone price distribution (samsung)',
                            filename= dirname + '/samsung_samrtphone_price_distribution.png')                                                       


    # Present Price profile of the head customers of Samsung Smartphones.    
    customer_id = df_smartphones_samsung['user_id'].value_counts()[:HEAD_CUSTOMERS_NUMBER].index

    fig, axes = plt.subplots(nrows=1, ncols=HEAD_CUSTOMERS_NUMBER, figsize=(15,10))
    for jj in range(HEAD_CUSTOMERS_NUMBER):
        customer = df_smartphones_samsung[df_smartphones_samsung['user_id'] == customer_id[jj]]
        plot_data_distribution( dataframe= customer,
                                col= 'price',
                                plot_type= 'violinplot',
                                ax=axes[jj],
                                title_str= 'Customer ' + str(customer_id[jj]),
                                filename= dirname + '/samsung_customers_price_profile.png')          