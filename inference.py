# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 10:00:28 2023

@author: 51027
"""

#this script is used to infer joint embedding for think aloud and desicison space
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
from preprocess import behavioral_embedding,text_embedding,str_to_number
from Data_loader import create_data_loaders
from model import JointEmbedding,TextDecisionModel
import matplotlib.pyplot as plt
import statsmodels.api as sm
from util import compute_similarity_distribution,decision_classifier,standardize,grouped_dimension_reduction,inference_data_preparation,visualize_data,min_max_normalize,plot_radar
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.cluster import KMeans
import csv
#prepare the trained model
# Define the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = JointEmbedding().to(device)
model = TextDecisionModel().to(device)

model_name = model.__class__.__name__ + '.pth'
model.load_state_dict(torch.load('../result/'+ model_name))
model.eval()

# #load the human dataset
behavioral_text_embedding_dataset = np.load('../result/behavioral_text_embeddings.npy', allow_pickle= True)
##or load GPT-4 sythetic dataset
# behavioral_text_embedding_dataset = np.load('../result/simulated_behavior_text_q_context_embeddings.npy', allow_pickle = True)
# behavioral_text_embedding_dataset = np.load('../result/simulated_behavior_text_q_embeddings.npy', allow_pickle = True)
behavioral_text_embedding_dataset = np.load('../result/musked_simulated_behavior_text_q_embeddings.npy', allow_pickle = True)

option_A_embedding, option_B_embedding, think_aloud_embeddings = inference_data_preparation(behavioral_text_embedding_dataset, list(range(12, 1548)))


compute_similarity_distribution(think_aloud_embeddings)
# np.random.shuffle(think_aloud_embeddings)

option_A_embedding = torch.tensor(option_A_embedding, device = device)
option_B_embedding = torch.tensor(option_B_embedding, device = device)
think_aloud_embeddings = torch.tensor(think_aloud_embeddings, device = device)


#infer think aloud and each option's shared embedding
with torch.no_grad():
    think_aloud_shared_embedding = model(think_aloud_embeddings)
    option_A_shared_embedding = option_A_embedding
    option_B_shared_embedding = option_B_embedding

    ##above is all you need for inference; below is our piprline for anlysis. You may refer to it or build you own pipeline.
    sim_to_A = torch.norm(think_aloud_shared_embedding - option_A_shared_embedding,dim=1)
    sim_to_B = torch.norm(think_aloud_shared_embedding - option_B_shared_embedding,dim=1)
    sim_to_A = think_aloud_shared_embedding - option_A_shared_embedding
    sim_to_B = think_aloud_shared_embedding - option_B_shared_embedding
    



##predicting choice probability based on a geometric perspective?   
#extract and plot histogram first
#when choosing A
sim_to_A = sim_to_A.cpu().numpy()
sim_to_B = sim_to_B.cpu().numpy()
choosing_A_idx = np.where(behavioral_text_embedding_dataset[:,2]==0)
choosing_B_idx = np.where(behavioral_text_embedding_dataset[:,2]==1)

# Plot the histograms
plt.hist(sim_to_A[choosing_A_idx], bins=50, alpha=0.5, label='Sim_to_A', color='blue')
plt.hist(sim_to_B[choosing_A_idx], bins=50, alpha=0.5, label='Sim_to_B', color='red')

# Labeling and showing the plot
plt.legend(loc='upper right')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Similarity in choosing A')
plt.grid(True)
plt.show()


# Plot the histograms
plt.hist(sim_to_A[choosing_B_idx], bins=50, alpha=0.5, label='Sim_to_A', color='blue')
plt.hist(sim_to_B[choosing_B_idx], bins=50, alpha=0.5, label='Sim_to_B', color='red')

# Labeling and showing the plot
plt.legend(loc='upper right')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Similarity in choosing B')
plt.grid(True)
plt.show()

#test statistically
choice = np.array(behavioral_text_embedding_dataset[:,2], dtype=int)
X = sim_to_B - sim_to_A  # your independent variable

# Add a constant to the independent variable
X = sm.add_constant(X)

# Fit the model
logit_model = sm.Logit(choice, X)
result = logit_model.fit()

# Print the summary
print(result.summary())


GPT_mean_acc_list = []
GPT_se_acc_list = []
##LOOCV
#use shared embedding distance to predict?
choice = np.array(behavioral_text_embedding_dataset[:,2], dtype =int)

##baseline model:  pca on text embeddings and then logistic regression (56%)
X = np.array(behavioral_text_embedding_dataset[:,10:1546],dtype = 'float32')
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)
# 对不同数量的主成分进行实验
max_components = 300  # 或者您想要尝试的最大主成分数
results = []
for n_components in range(1, max_components + 1):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_standardized)

    mean_acc, se_acc = decision_classifier(X_pca, choice, model='SVM')
    results.append((n_components, mean_acc, se_acc))
    print(f"主成分数: {n_components}, 平均准确率: {mean_acc:.2f}, 标准误: {se_acc:.2f}")

# Assuming 'results' is the list containing your data
header = ['Number of Components', 'Mean Accuracy', 'Standard Error']

# Specify the filename
filename = "result/pca_results.csv"

# Write data to CSV
with open(filename, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(header)  # write the header
    writer.writerows(results)  # write the data rows


# Read the CSV file into a DataFrame
results = pd.read_csv(filename)  

# Assuming results is your list of tuples
components = results['Number of Components']
mean_accs = results['Mean Accuracy']
se_accs = results['Standard Error']

plt.figure(figsize=(10, 6))
plt.plot(components, mean_accs, '-o', label='Mean LOOCV Accuracy')
plt.fill_between(components, 
                 [m - s for m, s in zip(mean_accs, se_accs)], 
                 [m + s for m, s in zip(mean_accs, se_accs)], 
                 color='gray', alpha=0.2)

plt.xlabel('Number of Components')
plt.ylabel('Mean LOOCV Accuracy')
plt.title('LOOCV Accuracy by Number of PCA Components')
plt.legend()
plt.show()
plt.savefig("result/pca_accuracy.png", format='png',bbox_inches='tight', dpi=600)

GPT_mean_acc_list.append(mean_acc)
GPT_se_acc_list.append(se_acc)

X = np.array(behavioral_text_embedding_dataset[:,12:1548],dtype = 'float32')
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)
pca = PCA(n_components=100)
X_pca = pca.fit_transform(X_standardized)
sum(pca.explained_variance_ratio_)

mean_acc, se_acc = decision_classifier(X_pca, choice, model='SVM')


#model 1: transformed think aloud embeddings (49%) (MLP:67%)
X = think_aloud_shared_embedding.cpu().numpy()
mean_acc, se_acc = logistic_regression(X,choice)
GPT_mean_acc_list.append(mean_acc)
GPT_se_acc_list.append(se_acc)

#model 2: a rough relative distance (computed by cosine similarity or Euclidean distance) (58.8%) (MLP:61.1%)
#infer think aloud and each option's shared embedding
with torch.no_grad():
    # think_aloud_shared_embedding, option_A_shared_embedding = model(think_aloud_embeddings, option_A_embedding)
    # think_aloud_shared_embedding, option_B_shared_embedding = model(think_aloud_embeddings, option_B_embedding)
    # sim_to_A = nn.functional.cosine_similarity(think_aloud_shared_embedding, option_A_shared_embedding)
    # sim_to_B = nn.functional.cosine_similarity(think_aloud_shared_embedding, option_B_shared_embedding)
    think_aloud_shared_embedding = model(think_aloud_embeddings)
    option_A_shared_embedding = option_A_embedding
    option_B_shared_embedding = option_B_embedding
    sim_to_A = torch.norm(think_aloud_shared_embedding - option_A_shared_embedding,dim=1)
    sim_to_B = torch.norm(think_aloud_shared_embedding - option_B_shared_embedding,dim=1)

    
sim_to_A = sim_to_A.cpu().numpy()
sim_to_B = sim_to_B.cpu().numpy()
X= np.array(sim_to_B-sim_to_A)
X = X.reshape((X.size, -1)) 
mean_acc, se_acc = decision_classifier(X,choice,model = 'Log')
GPT_mean_acc_list.append(mean_acc)
GPT_se_acc_list.append(se_acc)

#model 3: relative distances on each dimension sepeaterely (therefore 16 regressors + intercept) (MLP:66.8%)
with torch.no_grad():
    # think_aloud_shared_embedding, option_A_shared_embedding = model(think_aloud_embeddings, option_A_embedding)
    # think_aloud_shared_embedding, option_B_shared_embedding = model(think_aloud_embeddings, option_B_embedding)
    # sim_to_A = nn.functional.cosine_similarity(think_aloud_shared_embedding, option_A_shared_embedding)
    # sim_to_B = nn.functional.cosine_similarity(think_aloud_shared_embedding, option_B_shared_embedding)
    think_aloud_shared_embedding = model(think_aloud_embeddings)
    option_A_shared_embedding = option_A_embedding
    option_B_shared_embedding = option_B_embedding
    
think_aloud_shared_embedding = think_aloud_shared_embedding.cpu().numpy()
option_A_shared_embedding = option_A_shared_embedding.cpu().numpy()
option_B_shared_embedding = option_B_shared_embedding.cpu().numpy()

#aligned with model training, all the output should be standarized to get meaningful results.
option_A_shared_embedding = min_max_normalize(option_A_shared_embedding,dim=0)
option_B_shared_embedding = min_max_normalize(option_B_shared_embedding,dim=0)
# think_aloud_shared_embedding = min_max_normalize(think_aloud_shared_embedding,dim=0)

sim_to_A = np.abs(think_aloud_shared_embedding - option_A_shared_embedding)
sim_to_B = np.abs(think_aloud_shared_embedding - option_B_shared_embedding)

X= np.array(think_aloud_shared_embedding)
X= np.array(sim_to_B-sim_to_A)

mean_acc, se_acc = decision_classifier(X,choice,model = 'SVM')
GPT_mean_acc_list.append(mean_acc)
GPT_se_acc_list.append(se_acc)


#Try decode the embedding with following perspectives:
#1. Can the model recover individual differences? (we have 4 types of people in GPT-4 generated think aloud)
#stadardized each dimension to get a comparable scale

#infer think aloud and each option's shared embedding
behavioral_text_embedding_dataset = np.load('../result/musked_simulated_behavior_text_q_embeddings.npy', allow_pickle = True)
option_A_embedding, option_B_embedding, think_aloud_embeddings = inference_data_preparation(behavioral_text_embedding_dataset, list(range(12, 1548)))
option_A_embedding = torch.tensor(option_A_embedding, device = device)
option_B_embedding = torch.tensor(option_B_embedding, device = device)
think_aloud_embeddings = torch.tensor(think_aloud_embeddings, device = device)


with torch.no_grad():
    # think_aloud_shared_embedding, option_A_shared_embedding = model(think_aloud_embeddings, option_A_embedding)
    # think_aloud_shared_embedding, option_B_shared_embedding = model(think_aloud_embeddings, option_B_embedding)
    # sim_to_A = nn.functional.cosine_similarity(think_aloud_shared_embedding, option_A_shared_embedding)
    # sim_to_B = nn.functional.cosine_similarity(think_aloud_shared_embedding, option_B_shared_embedding)
    think_aloud_shared_embedding = model(think_aloud_embeddings)
    option_A_shared_embedding = option_A_embedding
    option_B_shared_embedding = option_B_embedding
    
think_aloud_shared_embedding = think_aloud_shared_embedding.cpu().numpy()
option_A_shared_embedding = option_A_shared_embedding.cpu().numpy()
option_B_shared_embedding = option_B_shared_embedding.cpu().numpy()

#aligned with model training, all the output should be standarized to get meaningful results.
option_A_shared_embedding = min_max_normalize(option_A_shared_embedding,dim=0)
option_B_shared_embedding = min_max_normalize(option_B_shared_embedding,dim=0)
think_aloud_shared_embedding = min_max_normalize(think_aloud_shared_embedding,dim=0)

sim_to_A = np.abs(think_aloud_shared_embedding - option_A_shared_embedding)
sim_to_B = np.abs(think_aloud_shared_embedding - option_B_shared_embedding)


distance_data = sim_to_B - sim_to_A
distance_data = think_aloud_shared_embedding


behavioral_distance_data = np.hstack([behavioral_text_embedding_dataset[:,1:12], distance_data])



data_column_names = ['sub_id','choice', 'p1','v1','p2','v2','problem_id','think_aloud_response','word_count','question_phrase', 'type',
                     'max_gain','min_gain','max_loss', 'min_loss', 'joint_max_median_gain','prob_max_gain','prob_min_gain','prob_max_loss','prob_min_loss','prob_joint_max_median_gain'
                     ,'Expected Utility','Entropy']
behavioral_distance_data = pd.DataFrame(behavioral_distance_data, columns = data_column_names)

distance_columns = ['max_gain','min_gain','max_loss', 'min_loss', 'joint_max_median_gain','prob_max_gain','prob_min_gain','prob_max_loss','prob_min_loss','prob_joint_max_median_gain'
                     ,'Expected Utility','Entropy']

selected_df = behavioral_distance_data[['type'] + distance_columns]

#for different types of individuals, do their think aloud really vary larger in some typical dimensions?
grouped_var = selected_df.groupby('type').var()
plot_radar(grouped_var, pic_name = 'GPT-4_variances')


##text embeddings pca and cluster
method = 'PCA'

individual_reduced_df = grouped_dimension_reduction(behavioral_text_embedding_dataset, 3, list(range(12, 1548)), method = method, by = 1, category = 11)


behavioral_joint_embedding_data = np.hstack([behavioral_text_embedding_dataset[:,1:12], sim_to_B-sim_to_A])
individual_reduced_df = grouped_dimension_reduction(behavioral_joint_embedding_data, 3, list(range(11, 27)), method = method, by = 0, category = 10)

# Convert the reduced_data to a DataFrame for easier handling
reduced_df = individual_reduced_df.copy()

visualize_data(reduced_df,'Category',n_dim =3)








##investigate on the empirical data
behavioral_text_embedding_dataset = np.load('../result/behavioral_text_embeddings.npy', allow_pickle= True)
option_A_embedding, option_B_embedding, think_aloud_embeddings = inference_data_preparation(behavioral_text_embedding_dataset, list(range(10, 1546)),mode = 'decision',query = 'local')

option_A_embedding = torch.tensor(option_A_embedding, device = device)
option_B_embedding = torch.tensor(option_B_embedding, device = device)
think_aloud_embeddings = torch.tensor(think_aloud_embeddings, device = device)

#infer think aloud and each option's shared embedding
with torch.no_grad():
    # think_aloud_shared_embedding, option_A_shared_embedding = model(think_aloud_embeddings, option_A_embedding)
    # think_aloud_shared_embedding, option_B_shared_embedding = model(think_aloud_embeddings, option_B_embedding)
    # sim_to_A = nn.functional.cosine_similarity(think_aloud_shared_embedding, option_A_shared_embedding)
    # sim_to_B = nn.functional.cosine_similarity(think_aloud_shared_embedding, option_B_shared_embedding)
    think_aloud_shared_embedding = model(think_aloud_embeddings)
    option_A_shared_embedding = option_A_embedding
    option_B_shared_embedding = option_B_embedding
    
    
think_aloud_shared_embedding = think_aloud_shared_embedding.cpu().numpy()
option_A_shared_embedding = option_A_shared_embedding.cpu().numpy()
option_B_shared_embedding = option_B_shared_embedding.cpu().numpy()

#align the option embedding as the neural network trained
option_A_shared_embedding[:,[0,1,2,3,4,10]] = option_A_shared_embedding[:,[0,1,2,3,4,10]]/1000
option_B_shared_embedding[:,[0,1,2,3,4,10]] = option_B_shared_embedding[:,[0,1,2,3,4,10]]/1000

#To evaluate variance contribution, all the dimensions should be standarized to get meaningful results.
option_A_shared_embedding = min_max_normalize(option_A_shared_embedding,dim=0)
option_B_shared_embedding = min_max_normalize(option_B_shared_embedding,dim=0)
think_aloud_shared_embedding = min_max_normalize(think_aloud_shared_embedding ,dim=0)


sim_to_A = np.abs(think_aloud_shared_embedding - option_A_shared_embedding)
sim_to_B = np.abs(think_aloud_shared_embedding - option_B_shared_embedding)


distance_data = sim_to_B - sim_to_A
distance_data = think_aloud_shared_embedding

behavioral_distance_data = np.hstack([behavioral_text_embedding_dataset[:,1:10], distance_data])



data_column_names = ['sub_id','choice', 'p1','v1','p2','v2','problem_id','think_aloud_response','word_count',
                     'max_gain','min_gain','max_loss', 'min_loss', 'joint_max_median_gain','prob_max_gain','prob_min_gain','prob_max_loss','prob_min_loss','prob_joint_max_median_gain'
                     ,'Expected Utility','Entropy']
behavioral_distance_data = pd.DataFrame(behavioral_distance_data, columns = data_column_names)

distance_columns = ['max_gain','min_gain','max_loss', 'min_loss', 'joint_max_median_gain','prob_max_gain','prob_min_gain','prob_max_loss','prob_min_loss','prob_joint_max_median_gain'
                     ,'Expected Utility','Entropy']

selected_df = behavioral_distance_data[['sub_id'] + distance_columns]
# selected_df['problem_id'] = selected_df['problem_id'].apply(lambda x: x[0])

individual_var = selected_df.groupby('sub_id').var()

K_range = range(1, 10)  # Example: trying from 1 to 10 clusters
sum_of_squared_distances = []
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42).fit(individual_var)
    sum_of_squared_distances.append(kmeans.inertia_)  # .inertia_ gives the sum of squared distances

# Step 3: Plot the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(K_range, sum_of_squared_distances, 'bx-')
plt.xlabel('k (Number of Clusters)')
plt.ylabel('Sum of Squared Distances')
plt.title('Elbow Method For Optimal k')
plt.show()

#K-mean cluster
# Initialize KMeans
kmeans = KMeans(n_clusters=5,random_state=42)  # specifying the number of clusters. In this example, it's 2.
kmeans.fit(individual_var[distance_columns])

# Getting the labels of each data point
labels = kmeans.labels_
labels = labels.reshape(-1,1)

#for different types of individuals, do their think aloud really vary larger in some typical dimensions?
individual_var['type'] = labels
type_counts = individual_var['type'].value_counts().sort_index()
plt.figure(figsize=(10, 6))
type_counts.plot(kind='bar')
plt.title('Counts of Observation Types')
plt.xlabel('Type')
plt.ylabel('Count')
plt.xticks(rotation=45)  # Rotates labels to make them readable
plt.show()

grouped_var = individual_var.groupby('type').mean()
plot_radar(grouped_var,'Human_variances')


Human_mean_acc_list = []
Human_se_acc_list = []
##LOOCV
#use shared embedding distance to predict?
choice = np.array(behavioral_text_embedding_dataset[:,2], dtype =int)

##baseline model:  pca on text embeddings and then logistic regression (56%)
X = np.array(behavioral_text_embedding_dataset[:,10:1546],dtype = 'float32')
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)
pca = PCA(n_components=12)

pca.fit(X_standardized)
X_pca = pca.transform(X_standardized)

mean_acc, se_acc = decision_classifier(X_pca,choice,model='SVM')
Human_mean_acc_list.append(mean_acc)
Human_se_acc_list.append(se_acc)


#model 1: transformed think aloud embeddings (MLP:57%)
X = think_aloud_shared_embedding
choice = np.array(behavioral_text_embedding_dataset[:,2], dtype =int)
mean_acc, se_acc = decision_classifier(X,choice)

Human_mean_acc_list.append(mean_acc)
Human_se_acc_list.append(se_acc)

#model 2: a rough relative distance (computed by cosine similarity or Euclidean distance)) (MLP:57.8%)
#infer think aloud and each option's shared embedding
with torch.no_grad():
    # think_aloud_shared_embedding, option_A_shared_embedding = model(think_aloud_embeddings, option_A_embedding)
    # think_aloud_shared_embedding, option_B_shared_embedding = model(think_aloud_embeddings, option_B_embedding)
    # sim_to_A = nn.functional.cosine_similarity(think_aloud_shared_embedding, option_A_shared_embedding)
    # sim_to_B = nn.functional.cosine_similarity(think_aloud_shared_embedding, option_B_shared_embedding)
    think_aloud_shared_embedding = model(think_aloud_embeddings)
    option_A_shared_embedding = model(option_A_embedding)
    option_B_shared_embedding = model(option_B_embedding)
    sim_to_A = torch.norm(think_aloud_shared_embedding - option_A_shared_embedding,dim=1)
    sim_to_B = torch.norm(think_aloud_shared_embedding - option_B_shared_embedding,dim=1)

    
sim_to_A = sim_to_A.cpu().numpy()
sim_to_B = sim_to_B.cpu().numpy()
X= np.array(sim_to_B-sim_to_A)
X = X.reshape((X.size, -1)) 
mean_acc, se_acc = decision_classifier(X,choice)
Human_mean_acc_list.append(mean_acc)
Human_se_acc_list.append(se_acc)

#model 3: relative distances on each dimension sepeaterely (therefore 12 regressors + intercept) (MLP:67.1%)
with torch.no_grad():
    # think_aloud_shared_embedding, option_A_shared_embedding = model(think_aloud_embeddings, option_A_embedding)
    # think_aloud_shared_embedding, option_B_shared_embedding = model(think_aloud_embeddings, option_B_embedding)
    # sim_to_A = think_aloud_shared_embedding - option_A_shared_embedding
    # sim_to_B = think_aloud_shared_embedding - option_B_shared_embedding
    think_aloud_shared_embedding = model(think_aloud_embeddings)
    option_A_shared_embedding = option_A_embedding
    option_B_shared_embedding = option_B_embedding
    

option_A_shared_embedding = option_A_shared_embedding.cpu().numpy()
option_B_shared_embedding = option_B_shared_embedding.cpu().numpy()
think_aloud_shared_embedding = think_aloud_shared_embedding.cpu().numpy()

#align the option embedding as the neural network trained
option_A_shared_embedding[:,[0,1,2,3,4,10]] = option_A_shared_embedding[:,[0,1,2,3,4,10]]/1000
option_B_shared_embedding[:,[0,1,2,3,4,10]] = option_B_shared_embedding[:,[0,1,2,3,4,10]]/1000

#aligned with model training, all the output should be standarized to get meaningful results.
option_A_shared_embedding = min_max_normalize(option_A_shared_embedding,dim=0)
option_B_shared_embedding = min_max_normalize(option_B_shared_embedding,dim=0)
# think_aloud_shared_embedding = min_max_normalize(think_aloud_shared_embedding,dim=0)

sim_to_A = np.abs(think_aloud_shared_embedding - option_A_shared_embedding)
sim_to_B = np.abs(think_aloud_shared_embedding - option_B_shared_embedding)

X= np.array(sim_to_B-sim_to_A)

mean_acc, se_acc = decision_classifier(X,choice,model = 'Log')
Human_mean_acc_list.append(mean_acc)
Human_se_acc_list.append(se_acc)

###adding another method
from transformers import pipeline

classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

dim_name = ['maximum-gain','minimum-gain','maximum-loss','minimum-loss','maxium-plus-median-gain',
            'maximum-gain probability','minimum-gain probability','maximum-loss probability',
            'minimum-loss probability','maxium-plus-median-gain probability','expected-utility','uncertainty']

opt_name = ['preferred', 'neutral', 'averse']

behavioral_text_q_data = np.load('result/behavioral_text_q_data.npy', allow_pickle = True)


for i in range(702,behavioral_text_q_data.shape[0]):
    tmp_context =  behavioral_text_q_data[i,10]
    tmp_think_aloud = 'Think Aloud response: ' + behavioral_text_q_data[i,8]
    tmp_prompt = tmp_context + tmp_think_aloud
    tmp_score = []
    for tmp_dim in dim_name:
        tmp_labels = []
        for tmp_opt in opt_name:
            tmp_labels.append(tmp_dim + ' ' + tmp_opt)
        output = classifier(tmp_prompt,
            candidate_labels = tmp_labels,
            multi_label=True,
        )
        # Initialize a dictionary to store the scores with key terms
        score_map = {'preferred': None, 'neutral': None, 'averse': None}
    
        # Loop through the labels and scores and map them
        for label, score in zip(output['labels'], output['scores']):
            if 'preferred' in label:
                score_map['preferred'] = score
            elif 'neutral' in label:
                score_map['neutral'] = score
            elif 'averse' in label:
                score_map['averse'] = score
        
        # Extract the scores in the desired order
        reordered_scores = [score_map[key] for key in ['preferred', 'neutral', 'averse']]
        tmp_score = tmp_score + reordered_scores
        
    if i ==0:
        dim_score = np.array(tmp_score)
    else:
        dim_score = np.vstack([dim_score,np.array(tmp_score)])

np.save('result/naive_machine_coding_score.npy',dim_score)

dim_score = np.load('result/naive_machine_coding_score.npy')
#model 4 naive machine coding to think aloud (0.644+_0.012)
choice = np.array(behavioral_text_q_data[:,2], dtype =int)
X = dim_score
mean_acc, se_acc = decision_classifier(X,choice,model='Log')



###load a word fraction model to predict choice
df_merged_fraction= pd.read_csv('../result/df_merged_fraction.csv')

# Define a mapping from category to integer (57%)
choice_mapping = {'A': 0, 'B': 1}
choice = df_merged_fraction['choice'].map(choice_mapping)
X = df_merged_fraction[['va','vb','pa','pb']]
X = np.column_stack([
    df_merged_fraction['vb'] - df_merged_fraction['va'],
    df_merged_fraction['pb'] - df_merged_fraction['pa']
])
mean_acc, se_acc = decision_classifier(X,choice,model='Log')