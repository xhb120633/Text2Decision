# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 16:32:44 2023

@author: 51027
"""

import numpy as np
import scipy.stats as stats
from sklearn.metrics.pairwise import cosine_similarity
import statsmodels.api as sm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
from preprocess import behavioral_embedding,text_embedding,str_to_number,option_prompt_generate,get_embeddings
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.svm import SVC


def compute_similarity_distribution(data):
    cosine_similarities = cosine_similarity(data)
    
    # Flatten the matrix and remove the lower triangle and diagonal 
    # (if you want to exclude identical embeddings)
    cosine_values = cosine_similarities[np.triu_indices(data.shape[0], k=1)]
    
    # If you want to include identical embeddings, then simply:
    # cosine_values = cosine_similarities.flatten()
    
    # Visualize the distribution
    import matplotlib.pyplot as plt
    plt.hist(cosine_values, bins=50, density=True)
    plt.title('Distribution of Cosine Similarities')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.show()
    
    # Calculate 95% confidence interval
    confidence_level = 0.95
    mean_val = np.mean(cosine_values)
    std_err = stats.sem(cosine_values)
    ci = stats.t.interval(confidence_level, len(cosine_values) - 1, loc=mean_val, scale=std_err)
    
    
    print(f"95% Confidence Interval: {ci}")

def decision_classifier(X, Y, Intercept = True, CV = 'LOOCV', model = 'Log'):
    n = len(Y)
    
    if model == 'Log':
        # Create logistic regression model
        log_reg = LogisticRegression()
        
        # Create leave-one-out cross-validator
        loo = LeaveOneOut()
        
        # Perform Leave-One-Out cross-validation
        accuracy_scores = cross_val_score(log_reg, X, Y, cv=loo)
        
        
    elif model == 'SVM':
        # Create SVM model
        svm_model = SVC()
        
        # Create leave-one-out cross-validator
        loo = LeaveOneOut()
        
        # Perform Leave-One-Out cross-validation
        accuracy_scores  = cross_val_score(svm_model, X, Y, cv=loo)        

    average_accuracy = np.mean(accuracy_scores)
    se_accuracy = np.std(accuracy_scores , ddof=1)/np.sqrt(n)
    print(f"Average Accuracy: {average_accuracy}")
    return average_accuracy, se_accuracy
    
    
def standardize(data, dim=0):
    """
    Standardize the input data along the specified dimension.
    
    :param data: Input data (numpy array or similar)
    :param dim: Dimension along which to standardize data (default is 0)
    :return: Standardized data
    """
    mean = np.mean(data, axis=dim, keepdims=True)
    std = np.std(data, axis=dim, keepdims=True)
    return (data - mean) / (std + 1e-7)  # Adding a small number to avoid division by zero

def min_max_normalize(data, dim=0):
    """
    Min-max normalize the input data along the specified dimension.
    
    :param data: Input data (numpy array or similar)
    :param dim: Dimension along which to normalize data (default is 0)
    :return: Min-max normalized data
    """
    min_val = np.min(data, axis=dim, keepdims=True)
    max_val = np.max(data, axis=dim, keepdims=True)
    return (data - min_val) / (max_val - min_val + 1e-7)  # Adding a small number to avoid division by zero



def grouped_dimension_reduction(data, n_components, original_dimensions, method, by, category):
    unique_subjects = np.unique(data[:, by])
    reduced_data_list = []
    category_list = []
    for subject in unique_subjects:
        subject_data = data[data[:, by] == subject]
        subject_category = subject_data[:,category][0]
        # Concatenating the data along the specified dimensions
        concatenated_data = subject_data[:, original_dimensions].reshape(1, -1)
        
        # Stacking the concatenated data of each subject
        if len(reduced_data_list) == 0:
            reduced_data_list = concatenated_data
        else:
            reduced_data_list = np.vstack((reduced_data_list, concatenated_data))
            
        category_list.append(subject_category)
        
    if method == 'PCA':
        reducer = PCA(n_components=n_components)
    elif method == 't-SNE':
        if n_components > 2:
            print("t-SNE usually used for 2D or 3D, but you can continue...")
        reducer = TSNE(n_components=n_components)
    else:
        raise ValueError("Invalid method. Use 'PCA' or 't-SNE'.")
    
    # Dimensionality reduction on the concatenated data
    
    reduced_data = reducer.fit_transform(reduced_data_list)
    column_names = [f'Dim{i}' for i in range(1, n_components + 1)]
    reduced_data = pd.DataFrame(reduced_data, columns = column_names)
    reduced_data['Category'] = category_list
    
    return reduced_data

def inference_data_preparation(data, text_embedding_index, mode = 'decision', query = 'local'):
    if mode == 'text':
        if query == 'local':
            option_A_embedding = np.load('../result/inference_option_A_embedding.npy')
            option_B_embedding = np.load('../result/inference_option_B_embedding.npy')
        elif query == 'online':
            model_name = 'text-embedding-ada-002'
            
            option_A_embedding = np.empty((1536))  
            option_B_embedding = np.empty((1536))  
            #obtain option behavioral embedding
            for i in range(data.shape[0]):
                tmp_data = data[i,:]
                #option A probability&outcome
                tmp_p = np.atleast_1d(str_to_number(tmp_data[3])/100)
                tmp_v = np.atleast_1d(str_to_number(tmp_data[4]))
                tmp_option_A_description = option_prompt_generate(tmp_p, tmp_v)
                tmp_option_A_embedding = get_embeddings(tmp_option_A_description,model_name)['data'][0]['embedding']
                option_A_embedding = np.vstack([option_A_embedding, tmp_option_A_embedding])
                
                tmp_p = np.atleast_1d((str_to_number(tmp_data[5])/100))
                tmp_v = np.atleast_1d(str_to_number(tmp_data[6]))
                tmp_option_B_description = option_prompt_generate(tmp_p, tmp_v)
                tmp_option_B_embedding = get_embeddings(tmp_option_B_description,model_name)['data'][0]['embedding']
                option_B_embedding = np.vstack([option_B_embedding, tmp_option_B_embedding])
                
            option_A_embedding = np.array(option_A_embedding, dtype = 'float32')
            option_B_embedding = np.array(option_B_embedding,dtype = 'float32')
            
            np.save('../result/inference_option_A_embedding.npy', option_A_embedding)
            np.save('../result/inference_option_B_embedding.npy', option_B_embedding)
    
    elif mode == 'decision':
        option_A_embedding = np.empty((12))  
        option_B_embedding = np.empty((12))  
        #obtain option behavioral embedding
        for i in range(data.shape[0]):
            tmp_data = data[i,:]
            #option A probability&outcome
            tmp_p = np.atleast_1d(str_to_number(tmp_data[3])/100)
            tmp_v = np.atleast_1d(str_to_number(tmp_data[4]))
            tmp_option_A_embedding = behavioral_embedding(tmp_p,tmp_v)
            option_A_embedding = np.vstack([option_A_embedding, tmp_option_A_embedding])
            
            tmp_p = np.atleast_1d((str_to_number(tmp_data[5])/100))
            tmp_v = np.atleast_1d(str_to_number(tmp_data[6]))
            tmp_option_B_embedding = behavioral_embedding(tmp_p,tmp_v)
            option_B_embedding = np.vstack([option_B_embedding, tmp_option_B_embedding])
        
        option_A_embedding = np.array(option_A_embedding[1:,:], dtype = 'float32')
        option_B_embedding = np.array(option_B_embedding[1:,:],dtype = 'float32')
    
    think_aloud_embeddings = np.array(data[:,text_embedding_index],dtype = 'float32')
    
    return option_A_embedding,  option_B_embedding, think_aloud_embeddings

def visualize_data(data, label, n_dim):
    """
    Visualize data in 2D or 3D scatter plot.
    
    Parameters:
    - data: DataFrame with data points.
    - label: Column name in the data representing the labels/categories.
    - n_dim: Number of dimensions (either 2 or 3).
    """

    # Set up the color palette
    sns.set_palette('husl', n_colors=len(data[label].unique()))

    if n_dim == 2:
        plt.figure(figsize=(10, 7))
        sns.scatterplot(x=data.columns[0], y=data.columns[1], hue=label, data=data)
        plt.title('2D Visualization')
        plt.show()

    elif n_dim == 3:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get unique labels and corresponding colors
        unique_labels = data[label].unique()
        colors = sns.color_palette('husl', n_colors=len(unique_labels))
        label_to_color = dict(zip(unique_labels, colors))
        
        for cat in unique_labels:
            subset = data[data[label] == cat]
            ax.scatter(subset[subset.columns[0]], subset[subset.columns[1]], subset[subset.columns[2]], 
                       label=cat, s=50, c=[label_to_color[cat]]*len(subset), depthshade=False)
        
        ax.set_xlabel(data.columns[0])
        ax.set_ylabel(data.columns[1])
        ax.set_zlabel(data.columns[2])
        ax.legend()
        plt.title('3D Visualization')
        plt.show()

    else:
        print("n_dim should be either 2 or 3")
        
def plot_radar(data,pic_name):
    """
    Generate a radar plot based on the given data.
    
    :param data: A dictionary containing data for the radar plot.
    """
    
    df = pd.DataFrame(data)

    # Number of variables (dimensions)
    categories = df.columns
    N = len(categories)

    # Calculate the angle of each axis in the plot
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # Set the figure and axes
    plt.figure(figsize=(15, 10))
    ax = plt.subplot(111, polar=True)

    # Set the first axis on top:
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axis per variable
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories,fontsize=15)

    # Plot data
    for index, row in df.iterrows():
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=index)

    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1),title = 'Type',prop={'size': 15}, title_fontsize='20')
    pic_name = pic_name + '.png'
    plt.savefig('../pic/Text2Decision/'+pic_name, bbox_inches='tight',dpi=600)  # 300 DPI is a common high-resolution setting
    plt.show()
