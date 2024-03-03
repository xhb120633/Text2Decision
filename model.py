# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 10:05:38 2023

@author: 51027
"""

import torch.nn as nn


class TextDecisionModel(nn.Module):
    def __init__(self, text_dim=1536, decision_dim=12):
        super(TextDecisionModel, self).__init__()
        
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, decision_dim)
        )

    def forward(self, text):
        return self.text_proj(text)