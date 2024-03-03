# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 10:22:15 2023

@author: 51027
"""

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

##2. construct dataloader, model for training
class JointDataset(Dataset):
    def __init__(self, text_embeddings, decision_embeddings, labels):
        self.text_embeddings = text_embeddings
        self.decision_embeddings = decision_embeddings
        self.labels = labels

        # Ensure both embeddings have the same number of samples
        assert self._get_num_samples(self.text_embeddings) == self._get_num_samples(self.decision_embeddings), "Mismatched number of samples between text and decision embeddings."

    def __len__(self):
        return self._get_num_samples(self.text_embeddings)

    def __getitem__(self, idx):
        return {
            "text_embedding": self.text_embeddings[idx],
            "decision_embedding": self.decision_embeddings[idx],
            "label": self.labels[idx]
        }

    def _get_num_samples(self, data):
        if isinstance(data, (list, tuple)):
            return len(data)
        elif isinstance(data, (np.ndarray, torch.Tensor)):
            return data.shape[0]
        else:
            raise TypeError("Unsupported type for embeddings")
            
def generate_labels(embeddings_1, embeddings_2, false_scale=1):
    # True labels
    true_labels = np.ones(embeddings_1.shape[0])
    
    # Initialize lists to hold the combined embeddings and labels
    combined_embeddings_1 = list(embeddings_1)
    combined_embeddings_2 = list(embeddings_2)
    combined_labels = list(true_labels)
    
    for _ in range(false_scale):
        # Shuffling embeddings_2 for false labels
        shuffled_embeddings_2 = np.random.permutation(embeddings_2)
        
        # Append false data and labels
        combined_embeddings_1.extend(embeddings_1)
        combined_embeddings_2.extend(shuffled_embeddings_2)
        combined_labels.extend(np.zeros(embeddings_1.shape[0]))  # false_labels
        
    # Convert lists back to numpy arrays
    combined_embeddings_1 = np.array(combined_embeddings_1)
    combined_embeddings_2 = np.array(combined_embeddings_2)
    combined_labels = np.array(combined_labels)
    
    return combined_embeddings_1, combined_embeddings_2, combined_labels  

class Dataset(Dataset):
    def __init__(self, text_embeddings, decision_embeddings):
        self.text_embeddings = text_embeddings
        self.decision_embeddings = decision_embeddings

        # Ensure both embeddings have the same number of samples
        assert self._get_num_samples(self.text_embeddings) == self._get_num_samples(self.decision_embeddings), "Mismatched number of samples between text and decision embeddings."

    def __len__(self):
        return self._get_num_samples(self.text_embeddings)

    def __getitem__(self, idx):
        return {
            "text_embedding": self.text_embeddings[idx],
            "decision_embedding": self.decision_embeddings[idx],
        }

    def _get_num_samples(self, data):
        if isinstance(data, (list, tuple)):
            return len(data)
        elif isinstance(data, (np.ndarray, torch.Tensor)):
            return data.shape[0]
        else:
            raise TypeError("Unsupported type for embeddings")
            
         
def create_data_loaders(text,decision, batch_size, test_size, contrastive = True, false_scale=1, scale = 'min_max'):
    ##generate true matching and mismatching labels
    if scale == 'min_max':
    # Initialize the scaler
        scaler = MinMaxScaler()
        
        # Fit and transform the data
        decision = scaler.fit_transform(decision)
    elif scale == 'outcome_scaling':
        decision[:,[0,1,2,3,4,10]] =  decision[:,[0,1,2,3,4,10]]/1000
    else:
        pass

        
    if contrastive:
        behavioral_embedding_dataset, text_problem_embeddings, label_dataset= generate_labels(decision, text, false_scale)
        label_dataset = np.array(label_dataset, dtype = 'float32')
    
        # Split into training and temporary sets (80% training, 20% temp)
        behavioral_embedding_train, behavioral_embedding_temp, text_problem_train, text_problem_temp, labels_train, labels_temp = train_test_split(behavioral_embedding_dataset, text_problem_embeddings, label_dataset, test_size=test_size, stratify=label_dataset)
    
        # Split the temporary set into validation and test sets (50% validation, 50% test of the remaining 20%)
        behavioral_embedding_val, behavioral_embedding_test, text_problem_val, text_problem_test, labels_val, labels_test = train_test_split(behavioral_embedding_temp, text_problem_temp, labels_temp, test_size=0.5, stratify=labels_temp)
    
        # First, we'll create Datasets for each split
        train_dataset = JointDataset(text_problem_train, behavioral_embedding_train, labels_train)
        val_dataset = JointDataset(text_problem_val, behavioral_embedding_val, labels_val)
        test_dataset = JointDataset(text_problem_test, behavioral_embedding_test, labels_test)
    else:
        
        # Split into training and temporary sets (80% training, 20% temp)
        behavioral_embedding_train, behavioral_embedding_temp, text_problem_train, text_problem_temp = train_test_split(decision, text, test_size=test_size)
    
        # Split the temporary set into validation and test sets (50% validation, 50% test of the remaining 20%)
        behavioral_embedding_val, behavioral_embedding_test, text_problem_val, text_problem_test = train_test_split(behavioral_embedding_temp, text_problem_temp, test_size=0.5)
    
        # First, we'll create Datasets for each split
        train_dataset = Dataset(text_problem_train, behavioral_embedding_train)
        val_dataset = Dataset(text_problem_val, behavioral_embedding_val)
        test_dataset = Dataset(text_problem_test, behavioral_embedding_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader 