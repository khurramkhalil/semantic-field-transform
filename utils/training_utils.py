#!/usr/bin/env python3
"""
Training utilities for SFT models
Modular training and optimization components
"""

import torch
import torch.nn as nn
import time
from sklearn.metrics import accuracy_score


class SFTTrainer:
    """
    Modular trainer for SFT models
    """
    
    def __init__(self, model, device='cpu', lr=1e-3, weight_decay=0.01):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        
    def train_epoch(self, train_loader, verbose=True):
        """
        Train for one epoch
        """
        self.model.train()
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            self.optimizer.zero_grad()
            
            inputs = batch['bytes'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass (returns output and evolved field)
            logits, evolved_field = self.model(inputs)
            loss = self.criterion(logits, labels)
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
            if verbose and batch_idx % 10 == 0:
                current_acc = total_correct / total_samples if total_samples > 0 else 0
                print(f"  Batch {batch_idx:3d}: Loss={loss.item():.4f}, Acc={current_acc:.4f}")
        
        epoch_time = time.time() - start_time
        epoch_acc = total_correct / total_samples
        epoch_loss = total_loss / len(train_loader)
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'time': epoch_time
        }
    
    def evaluate(self, test_loader):
        """
        Evaluate model on test set
        """
        self.model.eval()
        
        test_predictions = []
        test_labels_list = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch['bytes'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits, evolved_field = self.model(inputs)
                loss = self.criterion(logits, labels)
                
                predictions = torch.argmax(logits, dim=1)
                
                test_predictions.extend(predictions.cpu().numpy())
                test_labels_list.extend(labels.cpu().numpy())
                total_loss += loss.item()
        
        test_accuracy = accuracy_score(test_labels_list, test_predictions)
        test_loss = total_loss / len(test_loader)
        
        return {
            'accuracy': test_accuracy,
            'loss': test_loss,
            'predictions': test_predictions,
            'labels': test_labels_list
        }
    
    def train(self, train_loader, test_loader=None, epochs=10, verbose=True):
        """
        Full training loop with optional validation
        """
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, epochs)
        training_history = []
        
        for epoch in range(epochs):
            # Train epoch
            train_metrics = self.train_epoch(train_loader, verbose=verbose)
            
            # Evaluate if test loader provided
            test_metrics = {}
            if test_loader is not None:
                test_metrics = self.evaluate(test_loader)
            
            # Update scheduler
            scheduler.step()
            
            # Record history
            epoch_data = {
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_accuracy': train_metrics['accuracy'],
                'train_time': train_metrics['time'],
                'lr': scheduler.get_last_lr()[0]
            }
            
            if test_metrics:
                epoch_data.update({
                    'test_loss': test_metrics['loss'],
                    'test_accuracy': test_metrics['accuracy']
                })
            
            training_history.append(epoch_data)
            
            # Print epoch summary
            if verbose:
                print(f"Epoch {epoch+1:2d}: "
                      f"Train Loss={train_metrics['loss']:.4f}, "
                      f"Train Acc={train_metrics['accuracy']:.4f}, "
                      f"Time={train_metrics['time']:.1f}s, "
                      f"LR={scheduler.get_last_lr()[0]:.2e}")
                
                if test_metrics:
                    print(f"         Test Loss={test_metrics['loss']:.4f}, "
                          f"Test Acc={test_metrics['accuracy']:.4f}")
        
        return training_history


def train_complete_sft(model, train_loader, epochs=10, lr=1e-3, device='cpu'):
    """
    Convenience function for training SFT model
    """
    trainer = SFTTrainer(model, device=device, lr=lr)
    return trainer.train(train_loader, epochs=epochs)


def evaluate_sft(model, test_loader, device='cpu'):
    """
    Convenience function for evaluating SFT model
    """
    trainer = SFTTrainer(model, device=device)
    return trainer.evaluate(test_loader)

import torch.nn.functional as F

class SFT_MLM_Trainer:
    """
    A specialized trainer for the Masked Language Model (MLM) relational task.
    This trainer uses a cosine similarity loss to teach the model to "fill in the blanks",
    forcing it to learn the relationships between concepts.
    """
    
    def __init__(self, model, device='cpu', lr=1e-3, weight_decay=0.01):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        # The loss function is now Cosine Embedding Loss, which tries to maximize similarity.
        # It's equivalent to minimizing 1 - cosine_similarity.
        self.criterion = nn.CosineEmbeddingLoss()
        
    def train_epoch(self, train_loader, verbose=True):
        """
        Train for one epoch on the relational prediction task.
        """
        self.model.train()
        
        total_loss = 0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            self.optimizer.zero_grad()
            
            # Unpack the batch from our new MLM dataset
            masked_input = batch['masked_input'].to(self.device)
            target_input = batch['target_input'].to(self.device)
            mask_positions = batch['mask_positions'].to(self.device) # [B, L]
            
            # --- The Core MLM Logic ---

            # 1. Get the model's prediction for the masked regions
            # We run the masked input through the model to get its evolved field
            _ , evolved_masked_field = self.model(masked_input) # [B, P, Nc, D]

            # 2. Get the "true" field representation from the original sentence
            with torch.no_grad(): # Don't train on the target generation
                # Setting model to eval mode temporarily inside no_grad context
                self.model.eval()
                _ , target_field = self.model(target_input) # [B, P, Nc, D]
                self.model.train() # Set it back to train mode

            # 3. Select the field vectors ONLY at the masked positions
            # We want to compare what the model *predicts* for the mask with what *should be* there.
            
            # Flatten the spatial dimension to easily select with the boolean mask
            batch_size, resolution, n_components, semantic_dim = evolved_masked_field.shape
            
            # Predicted vectors at masked locations
            predicted_vectors = evolved_masked_field.view(batch_size * resolution, -1)[mask_positions.view(-1)]
            
            # Target vectors at the same masked locations
            target_vectors = target_field.view(batch_size * resolution, -1)[mask_positions.view(-1)]

            # We need to ensure we have something to compare
            if predicted_vectors.shape[0] == 0:
                continue

            # 4. Calculate the loss
            # We want the similarity to be 1 (hence `torch.ones`).
            # The loss will be low when the cosine similarity is high.
            target_similarity = torch.ones(predicted_vectors.shape[0]).to(self.device)
            loss = self.criterion(predicted_vectors, target_vectors, target_similarity)
            
            # --- End of Core Logic ---
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if verbose and batch_idx % 20 == 0:
                print(f"  Batch {batch_idx:3d}: MLM Loss={loss.item():.4f}")
        
        epoch_time = time.time() - start_time
        epoch_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
        
        return {
            'loss': epoch_loss,
            'time': epoch_time
        }

    def train(self, train_loader, epochs=10, verbose=True):
        """
        Full training loop for the MLM task.
        """
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, epochs)
        training_history = []
        
        print(f"Starting MLM training for {epochs} epochs...")
        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_loader, verbose=verbose)
            scheduler.step()
            
            epoch_data = {
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_time': train_metrics['time'],
                'lr': scheduler.get_last_lr()[0]
            }
            training_history.append(epoch_data)
            
            if verbose:
                print(f"Epoch {epoch+1:2d}: "
                      f"Train Loss={train_metrics['loss']:.4f}, "
                      f"Time={train_metrics['time']:.1f}s, "
                      f"LR={scheduler.get_last_lr()[0]:.2e}")
        
        return training_history


import torch.nn.functional as F

class SFT_Contrastive_Trainer:
    """
    The definitive trainer for SFT v2, using a contrastive (InfoNCE) loss.
    This approach prevents representational collapse by forcing the model to distinguish
    between a "positive" target and many "negative" distractors.
    """
    
    def __init__(self, model, device='cpu', lr=1e-3, weight_decay=0.01, temperature=0.07):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        # Temperature is a key hyperparameter in contrastive learning
        self.temperature = temperature
        
    def info_nce_loss(self, features):
        """
        Calculates the InfoNCE loss for a batch of features.
        Assumes the batch is structured as [pred_1, target_1, pred_2, target_2, ...].
        """
        # Reshape to easily pair predictions with their positive targets
        # [2B, D] -> [B, 2, D]
        reshaped_features = features.view(-1, 2, features.shape[-1])
        
        # Normalize features for stable cosine similarity calculation
        features = F.normalize(reshaped_features, dim=-1)
        
        # Separate predictions and targets
        predictions = features[:, 0]  # [B, D]
        targets = features[:, 1]      # [B, D]
        
        # Create the similarity matrix between all predictions and all targets
        # This will contain positive (diagonal) and negative (off-diagonal) pairs
        similarity_matrix = torch.matmul(predictions, targets.T) / self.temperature
        
        # The labels for cross_entropy are the indices of the positive pairs.
        # For our setup, this is just [0, 1, 2, ..., B-1].
        labels = torch.arange(predictions.shape[0]).long().to(self.device)
        
        # Calculate the loss. PyTorch's cross_entropy combines the log_softmax
        # and the negative log likelihood loss in one step.
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss

    def train_epoch(self, train_loader, verbose=True):
        """
        Train for one epoch on the contrastive relational prediction task.
        """
        self.model.train()
        
        total_loss = 0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            self.optimizer.zero_grad()
            
            masked_input = batch['masked_input'].to(self.device)
            target_input = batch['target_input'].to(self.device)
            
            # We need to process both masked and target inputs to get their fields
            # Concatenate them to run through the model in one go
            # Batch becomes [masked_1, target_1, masked_2, target_2, ...]
            combined_input = torch.cat([masked_input, target_input], dim=0)
            
            # The order in the batch needs to be interleaved for the loss function
            # Indices: [0_m, 1_m, ..., 0_t, 1_t, ...] -> [0_m, 0_t, 1_m, 1_t, ...]
            batch_size = masked_input.shape[0]
            indices = torch.arange(2 * batch_size).view(2, batch_size).t().flatten()
            interleaved_input = combined_input[indices]

            # --- The Core Contrastive Logic ---

            # 1. Run the entire interleaved batch through the model
            _ , evolved_fields = self.model(interleaved_input)
            
            # # 2. Pool the fields to get a single vector for each sentence
            # pooled_vectors = evolved_fields.view(evolved_fields.shape[0], -1).mean(dim=1)
            
            # --- AFTER (Corrected Code) ---
            # 2. Pool the fields to get a single vector for each sentence
            # First, average across the positional dimension (P)
            # Shape: [B, P, Nc, D] -> [B, Nc, D]
            pooled_across_positions = evolved_fields.mean(dim=1)
            # Then, flatten the component and semantic dimensions to get our final vector
            # Shape: [B, Nc, D] -> [B, Nc * D]
            pooled_vectors = pooled_across_positions.view(pooled_across_positions.shape[0], -1)            
            
            # 3. Calculate the InfoNCE loss
            loss = self.info_nce_loss(pooled_vectors)
            
            # --- End of Core Logic ---
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if verbose and batch_idx % 20 == 0:
                print(f"  Batch {batch_idx:3d}: Contrastive Loss={loss.item():.4f}")
        
        epoch_time = time.time() - start_time
        epoch_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
        
        return {
            'loss': epoch_loss,
            'time': epoch_time
        }

    def train(self, train_loader, epochs=10, verbose=True):
        """
        Full training loop for the contrastive MLM task.
        """
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, epochs)
        training_history = []
        
        print(f"Starting Contrastive training for {epochs} epochs...")
        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_loader, verbose=verbose)
            scheduler.step()
            
            epoch_data = {
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_time': train_metrics['time'],
                'lr': scheduler.get_last_lr()[0]
            }
            training_history.append(epoch_data)
            
            if verbose:
                print(f"Epoch {epoch+1:2d}: "
                      f"Train Loss={train_metrics['loss']:.4f}, "
                      f"Time={train_metrics['time']:.1f}s, "
                      f"LR={scheduler.get_last_lr()[0]:.2e}")
        
        return training_history
