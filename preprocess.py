# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 10:42:04 2023

@author: 51027
"""

import pandas as pd
import numpy as np
import json
import random
import openai

openai.api_key = 'xxxxxx' #add you own API key
# 1. data preparation. Load choice13k problems
def preprocess_data(data):
    
    df = pd.DataFrame(data)

    # Melt the DataFrame
    df_melted = df.melt(ignore_index=False, value_name='data')
    
    # Extract p and v values
    df_melted['p'] = df_melted['data'].apply(lambda x: [item[0] for item in x]).apply(lambda x: [round(item, 4) for item in x])
    df_melted['v'] = df_melted['data'].apply(lambda x: [item[1] for item in x])
    
    # Merge duplicate v values and sum their probabilities
    def merge_v_p(v_list, p_list):
        d = {}
        for v, p in zip(v_list, p_list):
            if v in d:
                d[v] += p
            else:
                d[v] = p
        return list(d.keys()), list(d.values())
    
    df_melted[['v', 'p']] = df_melted.apply(lambda row: merge_v_p(row['v'], row['p']), axis=1, result_type='expand')
    
    # Drop unnecessary columns and reset index
    df_final = df_melted.drop(columns=['variable', 'data']).reset_index(drop=True)
    
    return df_final

def option_prompt_generate(p,v):
    option = ''

    for i in range(len(p)):
        if len(p)== 1:
            option = str(v[0]) + ' dollars with ' + str(p[0]*100) + ' % chance'
        else:
            option = option+ str(v[i]) + ' dollars with ' + str(p[i]*100) + ' % chance'
            
        if i == len(p)-1 :
            punc = '.'
        else:
            punc = ', '
        option = option + punc
    
    return(option)

def get_embeddings(text, model): 
    response = openai.Embedding.create(
            input=text,
            engine= model) 
    return response


def behavioral_embedding(p,v):
    p = np.array(p,dtype = 'float')
    v = np.array(v, dtype = 'float')
    ##Heuristics
    # maximum gain
    if np.all(v<=0):
        max_gain = 0
    else:
        max_gain = v[v == np.max(v)]
    # minimum gain
    if np.all(v<=0):
        min_gain = 0
    else:
        v_positive = v[v>=0]
        min_gain = v[v == np.min(v_positive)]
    # maximum loss
    if np.all(v>=0):
        max_loss = 0
    else:
        max_loss = v[v == np.min(v)]
    # minimum loss
    if np.all(v>=0):
        min_loss = 0
    else:
        v_negative = v[v<=0]
        min_loss = v[v == np.max(v_negative)]
    # joint max_gain and median gain (the second high)
    if v.size == 1:
       if v>0:
           joint_max_median_gain = v
       else:
           joint_max_median_gain = 0
    else:
        sorted_v = np.sort(v)[::-1]
        if sum(sorted_v>=0)>1:
            joint_max_median_gain = sorted_v[0] + sorted_v[1]
        else:
            joint_max_median_gain = 0
            
    # probability of maximum gain
    if np.all(v<=0):
        prob_max_gain = 0
    else:
        prob_max_gain = p[v == np.max(v)] 
    # probability of minimum gain
    if np.all(v<=0):
        prob_min_gain = 0
    else:
        v_positive = v[v>=0]
        prob_min_gain = p[v == np.min(v_positive)]
    # probability of maximum loss
    if np.all(v>=0):
        prob_max_loss = 0
    else:
        prob_max_loss = p[v == np.min(v)]
    # minimum loss
    if np.all(v>=0):
        prob_min_loss = 0
    else:
        v_negative = v[v<=0]
        prob_min_loss = p[v == np.max(v_negative)]
    # joint probability of max_gain and median gain (the second high)
    if v.size == 1:
       if v>0:
           prob_joint_max_median_gain = p
       else:
           prob_joint_max_median_gain = 0
    else:
        sorted_v = np.sort(v)[::-1]
        if sum(sorted_v>=0)>1:
            prob_joint_max_median_gain = p[v==sorted_v[0]] + p[v==sorted_v[1]]
        else:
            prob_joint_max_median_gain = 0
            
        
    ####Normative
    EV = np.dot(v,p)
    if p.size == 1:
        H = 0
    else:
        H = -sum(p*np.log2(p))
    
    behavioral_embedding=np.array([max_gain,min_gain,max_loss, min_loss, joint_max_median_gain,prob_max_gain,prob_min_gain,prob_max_loss,prob_min_loss,prob_joint_max_median_gain,EV,H],dtype ='object')
    behavioral_embedding = np.hstack(behavioral_embedding)
    return(behavioral_embedding)


def behavioral_embedding_model(behavioral_data):
    behavioral_data = preprocess_data(behavioral_data)
    behavioral_embedding_dataset = np.empty((12))  
    for i in range(behavioral_data.shape[0]):
        tmp_data = behavioral_data.iloc[i,:]       
        tmp_behavioral_embedding = behavioral_embedding(tmp_data['p'],tmp_data['v'])
        behavioral_embedding_dataset = np.vstack([behavioral_embedding_dataset, tmp_behavioral_embedding])
        
    behavioral_embedding_dataset = behavioral_embedding_dataset[1:,]
    behavioral_embedding_dataset = np.array(behavioral_embedding_dataset,dtype='float32')
    return behavioral_embedding_dataset

def text_embedding(behavioral_data, query):
    if query == 'local':
        text_problem_embeddings = np.load('../result/c13k_problem_embeddings.npy')
    elif query == 'online':
        prompt_dataset = np.empty((0,1))   
        for i in range(behavioral_data.shape[0]):
            tmp_data = behavioral_data.iloc[i,:]
            tmp_prompt = option_prompt_generate(tmp_data['p'],tmp_data['v'])
            tmp_prompt = np.array(tmp_prompt, dtype ='object')
            prompt_dataset = np.vstack([prompt_dataset,tmp_prompt])
            
        model_name = 'text-embedding-ada-002'
        text_problem_embeddings = np.empty((prompt_dataset.shape[0],1536))
        for i in range(prompt_dataset.shape[0]):
            tmp_embeddings = get_embeddings(prompt_dataset[i,0],model_name)
            text_problem_embeddings [i,:] = tmp_embeddings['data'][0]['embedding']
    
    text_problem_embeddings = np.array(text_problem_embeddings,dtype='float32')
    return(text_problem_embeddings)


def str_to_number(s):
    #s should be a series from dataframe
    tmp_chr = str(s)
    #remove brackets
    tmp_chr = tmp_chr[1:-1]
    #split the str to numbers
    strlist = tmp_chr.split(',')
    for i in range(len(strlist)):
        if i == 0:
            A = np.array(float(strlist[i]))
        else:
            A = np.hstack([A,float(strlist[i])])
    return(A)

