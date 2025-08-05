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

def create_compositional_dataset():
    """
    Creates a highly controlled dataset to test for semantic composition.

    - Training set: Contains sentences about color OR shape, but never both.
    - Test set: Contains sentences combining color AND shape.
    - Basis set: Contains sentences representing "pure" concepts.

    Returns:
        dict: A dictionary containing 'train', 'test', and 'basis' sets.
    """
    print("Creating compositional dataset...")

    # Define concepts
    colors = {"red": ["ball", "car"], "blue": ["block", "pyramid"]}
    shapes = {"sphere": ["round thing"], "cube": ["square box"]}
    
    # --- Training Data (Color OR Shape) ---
    train_texts = []
    train_labels = []

    # Color sentences (label 0)
    for color, objects in colors.items():
        for obj in objects:
            train_texts.append(f"The {obj} is {color}")
            train_texts.append(f"I see a {color} {obj}")
    train_labels.extend([0] * len(train_texts))

    # Shape sentences (label 1)
    shape_texts_start_index = len(train_texts)
    for shape, objects in shapes.items():
        for obj in objects:
            train_texts.append(f"Look at the {obj}")
            train_texts.append(f"It is a {shape}")
    train_labels.extend([1] * (len(train_texts) - shape_texts_start_index))

    # --- Test Data (Sentences NEVER seen in training) ---
    test_texts = []
    for shape in shapes.keys():
        for color in colors.keys():
            test_texts.append(f"The {shape} is {color}")
            test_texts.append(f"A {color} {shape}")

    # --- Basis Data (For comparing learned concepts) ---
    basis_texts = []
    basis_concepts = []

    for color in colors.keys():
        basis_texts.append(f"The concept is {color}")
        basis_concepts.append(color)

    for shape in shapes.keys():
        basis_texts.append(f"The concept is a {shape}")
        basis_concepts.append(shape)
        
    compositional_dataset = {
        "train": {"texts": train_texts, "labels": train_labels},
        "test": {"texts": test_texts, "labels": [0] * len(test_texts)},  # Labels don't matter for test
        "basis": {"texts": basis_texts, "concepts": basis_concepts}
    }

    print(f"  Training samples: {len(train_texts)}")
    print(f"  Test samples (unseen compositions): {len(test_texts)}")
    print(f"  Basis concepts: {len(basis_texts)}")
    
    return compositional_dataset

def create_svo_compositional_dataset():
    """
    Creates a dataset to test Subject-Verb-Object role understanding.

    - Training set: Contains sentences about agents OR actions on objects.
    - Test set: Contains full SVO sentences, including role-swapped versions.
    - Basis set: Contains sentences representing "pure" concepts for analysis.

    Returns:
        dict: A dictionary containing 'train', 'test', and 'basis' sets.
    """
    print("Creating SVO compositional dataset...")

    # Define concepts
    agents = ["The dog", "The cat", "The boy", "The girl"]
    verbs = ["chased", "threw", "saw", "pushed"]
    objects = ["the ball", "the stick", "the toy", "the box"]

    # --- Training Data (Agent OR Action/Object) ---
    train_texts = []
    train_labels = []

    # Agent sentences (label 0)
    for agent in agents:
        train_texts.append(f"{agent} runs")
        train_texts.append(f"{agent} sleeps")
    train_labels.extend([0] * len(train_texts))

    # Action/Object sentences (label 1)
    action_texts_start_index = len(train_texts)
    for verb in verbs:
        for obj in objects:
            # Only add a subset to avoid too large a training set
            if (verbs.index(verb) + objects.index(obj)) % 2 == 0:
                train_texts.append(f"{verb} {obj}")
    train_labels.extend([1] * (len(train_texts) - action_texts_start_index))

    # --- Test Data (Unseen SVO compositions) ---
    test_texts = [
        "The dog chased the ball",
        "The cat threw the stick",
        "The girl saw the toy",
        "The boy pushed the box",
        # Crucial inverted test cases
        "The ball chased the dog",
        "The stick threw the cat",
    ]

    # --- Basis Data (For comparing learned concepts) ---
    basis_texts = []
    basis_concepts = []
    
    for concept in agents + verbs + objects:
        basis_texts.append(f"Concept is {concept}")
        basis_concepts.append(concept)
        
    compositional_dataset = {
        "train": {"texts": train_texts, "labels": train_labels},
        "test": {"texts": test_texts, "labels": [0] * len(test_texts)},
        "basis": {"texts": basis_texts, "concepts": basis_concepts}
    }

    print(f"  Training samples: {len(train_texts)}")
    print(f"  Test samples (unseen compositions): {len(test_texts)}")
    print(f"  Basis concepts: {len(basis_texts)}")
    
    return compositional_dataset

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


# Define a special byte value for the MASK token that is unlikely to appear in text
MASK_TOKEN_ID = 254 

def create_mlm_dataset():
    """
    Creates a dataset for the Masked Language Modeling (MLM) / relational task.
    For each sentence, it generates multiple versions with different parts masked.
    
    Returns:
        A list of training tuples: (masked_sentence, original_sentence)
    """
    print("Creating Masked Language Model (MLM) dataset...")

    # Define concepts
    agents = ["The dog", "The cat", "The boy", "The girl"]
    verbs = ["chased", "threw", "saw", "pushed"]
    objects = ["the ball", "the stick", "the toy", "the box"]

    # Generate all possible SVO sentences
    full_sentences = []
    for agent in agents:
        for verb in verbs:
            for obj in objects:
                full_sentences.append(f"{agent} {verb} {obj}")

    training_pairs = []
    for sentence in full_sentences:
        parts = sentence.split()
        agent, verb, obj = parts[0:2], [parts[2]], parts[3:5]
        
        # Re-join with spaces to handle multi-word concepts
        agent_str = " ".join(agent)
        verb_str = " ".join(verb)
        obj_str = " ".join(obj)

        # 1. Mask the object
        masked_obj_sent = f"{agent_str} {verb_str} [MASK]"
        training_pairs.append((masked_obj_sent, sentence))
        
        # 2. Mask the verb
        masked_verb_sent = f"{agent_str} [MASK] {obj_str}"
        training_pairs.append((masked_verb_sent, sentence))

        # 3. Mask the agent
        masked_agent_sent = f"[MASK] {verb_str} {obj_str}"
        training_pairs.append((masked_agent_sent, sentence))

    print(f"  Generated {len(full_sentences)} base sentences.")
    print(f"  Created {len(training_pairs)} masked training pairs.")
    
    return training_pairs


class SFT_MLM_Dataset(Dataset):
    """
    Dataset specifically for the Masked Language Modeling task.
    Returns a pair: the masked input and the original target.
    """
    def __init__(self, training_pairs, max_length=64):
        self.training_pairs = training_pairs
        self.max_length = max_length
        self.mask_token_str = "[MASK]"

    def __len__(self):
        return len(self.training_pairs)

    def _sentence_to_tensor(self, sentence, mask_positions=False):
        """Helper to convert a sentence string to a byte tensor and find mask positions."""
        # Replace [MASK] string with a placeholder before encoding
        # We find its position first, then replace
        mask_indices = []
        if mask_positions:
            start_index = sentence.find(self.mask_token_str)
            if start_index != -1:
                # Find the indices covered by the mask string
                mask_len = len(self.mask_token_str.encode('utf-8', errors='ignore'))
                for i in range(mask_len):
                    mask_indices.append(start_index + i)
        
        # Encode the sentence, using our special MASK_TOKEN_ID
        sentence_with_token = sentence.replace(self.mask_token_str, chr(MASK_TOKEN_ID) * len(self.mask_token_str))
        byte_seq = sentence_with_token.encode('utf-8', errors='ignore')[:self.max_length]
        
        padded = np.zeros(self.max_length, dtype=np.int64)
        padded[:len(byte_seq)] = list(byte_seq)
        
        mask_tensor = torch.zeros(self.max_length, dtype=torch.bool)
        mask_tensor[mask_indices] = True
        
        return torch.tensor(padded, dtype=torch.long), mask_tensor

    def __getitem__(self, idx):
        masked_sentence, original_sentence = self.training_pairs[idx]
        
        # Convert masked sentence and get mask positions
        masked_tensor, mask_positions = self._sentence_to_tensor(masked_sentence, mask_positions=True)
        
        # Convert original sentence (no need for mask positions)
        original_tensor, _ = self._sentence_to_tensor(original_sentence)

        return {
            'masked_input': masked_tensor,
            'target_input': original_tensor,
            'mask_positions': mask_positions, # Tensor of bools indicating where the mask is
            'masked_text': masked_sentence,
            'original_text': original_sentence
        }

def create_svo_compositional_dataset():
    """
    Creates a dataset to test Subject-Verb-Object role understanding.
    (This is the new function for the SVO experiment)
    """
    print("Creating SVO compositional dataset...")

    agents = ["The dog", "The cat", "The boy", "The girl"]
    verbs = ["chased", "threw", "saw", "pushed"]
    objects = ["the ball", "the stick", "the toy", "the box"]

    # Training Data (Agent OR Action/Object)
    train_texts, train_labels = [], []
    for agent in agents:
        train_texts.extend([f"{agent} runs", f"{agent} sleeps"])
    train_labels.extend([0] * len(train_texts))

    action_texts_start_index = len(train_texts)
    for verb in verbs:
        for obj in objects:
            if (verbs.index(verb) + objects.index(obj)) % 2 == 0:
                train_texts.append(f"{verb} {obj}")
    train_labels.extend([1] * (len(train_texts) - action_texts_start_index))

    # Test Data (Unseen SVO compositions)
    test_texts = [
        "The dog chased the ball", "The cat threw the stick",
        "The girl saw the toy", "The boy pushed the box",
        "The ball chased the dog", "The stick threw the cat",
    ]

    # Basis Data (For comparing learned concepts)
    basis_texts, basis_concepts = [], []
    for concept in agents + verbs + objects:
        basis_texts.append(f"Concept is {concept}")
        basis_concepts.append(concept)
        
    compositional_dataset = {
        "train": {"texts": train_texts, "labels": train_labels},
        "test": {"texts": test_texts, "labels": [0] * len(test_texts)},
        "basis": {"texts": basis_texts, "concepts": basis_concepts}
    }

    print(f"  Training samples: {len(train_texts)}")
    print(f"  Test samples: {len(test_texts)}")
    print(f"  Basis concepts: {len(basis_concepts)}")
    
    return compositional_dataset


def create_data_loaders(texts, labels, batch_size=8, test_split=0.2, max_length=64):
    """
    Create train/test data loaders from texts and labels
    """
    split_idx = int((1 - test_split) * len(texts))
    train_texts, test_texts = texts[:split_idx], texts[split_idx:]
    train_labels, test_labels = labels[:split_idx], labels[split_idx:]
    
    train_dataset = SFTDataset(train_texts, train_labels, max_length)
    test_dataset = SFTDataset(test_texts, test_labels, max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def text_to_tensor(text, max_length=64, device='cpu'):
    """
    Convert text to byte tensor for model input
    """
    byte_seq = text.encode('utf-8')[:max_length]
    padded = np.zeros(max_length, dtype=np.int64)
    padded[:len(byte_seq)] = list(byte_seq)
    return torch.tensor(padded, dtype=torch.long).unsqueeze(0).to(device)
