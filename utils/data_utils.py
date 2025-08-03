#!/usr/bin/env python3
"""
Data utilities for SFT experiments
Modular data handling and dataset creation
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class SFTDataset(Dataset):
    """
    Flexible dataset for SFT experiments
    """
    def __init__(self, texts, labels, max_length=64):
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        byte_seq = text.encode('utf-8', errors='ignore')[:self.max_length]
        padded = np.zeros(self.max_length, dtype=np.int64)
        padded[:len(byte_seq)] = list(byte_seq)
        
        return {
            'bytes': torch.tensor(padded, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long),
            'text': text
        }


def create_test_datasets():
    """
    Create comprehensive test datasets for validation
    """
    # Designed test cases to validate different aspects
    test_texts = [
        # Word order sensitivity tests
        "The quick brown fox jumps over the lazy dog",
        "The lazy dog jumps over the quick brown fox",
        "Fox brown quick the over jumps dog lazy the",
        
        # Semantic similarity tests  
        "I love this amazing movie with great acting",
        "I hate this terrible movie with awful acting",
        "This movie has incredible acting and amazing story",
        
        # Syntactic structure tests
        "The cat sat on the mat",
        "On the mat sat the cat", 
        "Cat the on mat the sat",
        
        # Length variation tests
        "Good",
        "Very good",
        "This is very good indeed",
        "This particular example is extremely good in every possible way"
    ] * 4  # Repeat for more training data
    
    # Create balanced labels
    labels = []
    for i in range(len(test_texts)):
        if any(word in test_texts[i] for word in ['love', 'amazing', 'great', 'good', 'incredible']):
            labels.append(1)  # Positive
        else:
            labels.append(0)  # Negative/Neutral
    
    return test_texts, labels


def create_data_loaders(texts, labels, batch_size=8, test_split=0.2, max_length=64):
    """
    Create train/test data loaders from texts and labels
    """
    # Train/test split
    split_idx = int((1 - test_split) * len(texts))
    train_texts, test_texts = texts[:split_idx], texts[split_idx:]
    train_labels, test_labels = labels[:split_idx], labels[split_idx:]
    
    # Create datasets
    train_dataset = SFTDataset(train_texts, train_labels, max_length)
    test_dataset = SFTDataset(test_texts, test_labels, max_length)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def create_validation_test_pairs():
    """
    Create test pairs for information preservation validation
    """
    return [
        ("The dog quickly chased the cat", "The dog slowly chased the cat"),
        ("I love this movie", "I hate this movie"),
        ("The algorithm is efficient", "The algorithm is inefficient"),
        ("Python programming", "Programming Python")
    ]


def create_localization_test_cases():
    """
    Create test cases for local information preservation tests
    """
    return [
        ("abc def ghi", [0.2, 0.5, 0.8]),  # Should show peaks at word positions
        ("programming language", [0.3, 0.7]),  # Two main semantic regions
        ("a", [0.5])  # Single character - sharp localization
    ]


def text_to_tensor(text, max_length=64, device='cpu'):
    """
    Convert text to byte tensor for model input
    """
    byte_seq = text.encode('utf-8')
    padded = np.zeros(max_length, dtype=np.int64)
    padded[:len(byte_seq)] = list(byte_seq)
    return torch.tensor(padded, dtype=torch.long).unsqueeze(0).to(device)