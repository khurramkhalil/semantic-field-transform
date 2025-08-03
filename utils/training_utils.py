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