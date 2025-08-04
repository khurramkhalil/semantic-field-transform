#!/usr/bin/env python3
"""
Visualization utilities for SFT models
Modular visualization and plotting components
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_utils import text_to_tensor

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_field_dynamics(model, text_samples, device='cpu', save_path='field_dynamics.png'):
    """
    Comprehensive visualization of the complete SFT process
    """
    print("\n📊 COMPREHENSIVE FIELD DYNAMICS VISUALIZATION")
    print("-" * 60)
    
    model.eval()
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    with torch.no_grad():
        for idx, text in enumerate(text_samples[:2]):
            # Convert to input
            byte_tensor = text_to_tensor(text, max_length=32, device=device)
            
            # Get positions
            positions = torch.linspace(0, 1, 128, device=device).unsqueeze(0)
            positions_np = positions.squeeze().cpu().numpy()
            
            # Get field evolution
            current_field = model.local_field.compute_local_continuous_field(byte_tensor, positions)
            
            # Plot initial field
            initial_field_np = current_field.squeeze(0).cpu().numpy()
            axes[0, idx].plot(positions_np, initial_field_np[:, 0], 'b-', linewidth=2, label='Dim 0')
            axes[0, idx].plot(positions_np, initial_field_np[:, 1], 'r-', linewidth=2, label='Dim 1')
            axes[0, idx].set_title(f'Initial Field: "{text[:25]}..."')
            axes[0, idx].set_ylabel('Field Amplitude')
            axes[0, idx].legend()
            axes[0, idx].grid(True, alpha=0.3)
            
            # Evolve and plot evolved field
            for evolution_layer in model.evolution_layers:
                evolved = evolution_layer(current_field)
                current_field = current_field + evolved
            
            evolved_field_np = current_field.squeeze(0).cpu().numpy()
            axes[1, idx].plot(positions_np, evolved_field_np[:, 0], 'b-', linewidth=2, label='Dim 0')
            axes[1, idx].plot(positions_np, evolved_field_np[:, 1], 'r-', linewidth=2, label='Dim 1')
            axes[1, idx].set_title(f'Evolved Field')
            axes[1, idx].set_ylabel('Field Amplitude')
            axes[1, idx].legend()
            axes[1, idx].grid(True, alpha=0.3)
            
            # Plot field energy distribution
            field_energy = torch.sum(current_field**2, dim=-1).squeeze(0).cpu().numpy()
            axes[2, idx].plot(positions_np, field_energy, 'g-', linewidth=3, label='Semantic Energy')
            axes[2, idx].fill_between(positions_np, field_energy, alpha=0.3, color='green')
            axes[2, idx].set_title(f'Semantic Energy Distribution')
            axes[2, idx].set_ylabel('Energy |Ψ(x)|²')
            axes[2, idx].set_xlabel('Continuous Position x')
            axes[2, idx].legend()
            axes[2, idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_training_curves(training_history, save_path='training_curves.png'):
    """
    Plot training curves and performance metrics
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = [h['epoch'] for h in training_history]
    train_losses = [h['train_loss'] for h in training_history]
    train_accuracies = [h['train_accuracy'] for h in training_history]
    times = [h['train_time'] for h in training_history]
    
    # Training loss
    ax1.plot(epochs, train_losses, 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Convergence')
    ax1.grid(True, alpha=0.3)
    
    # Training accuracy
    ax2.plot(epochs, train_accuracies, 'g-o', linewidth=2, markersize=6)
    
    # Add test accuracy if available
    if 'test_accuracy' in training_history[0]:
        test_accuracies = [h['test_accuracy'] for h in training_history]
        ax2.plot(epochs, test_accuracies, 'r-o', linewidth=2, markersize=6, label='Test Acc')
        ax2.legend()
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Progress')
    ax2.grid(True, alpha=0.3)
    
    # Training time per epoch
    ax3.plot(epochs, times, 'r-o', linewidth=2, markersize=6)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Time (seconds)')
    ax3.set_title('Training Time per Epoch')
    ax3.grid(True, alpha=0.3)
    
    # Final performance summary
    final_train_acc = train_accuracies[-1]
    final_test_acc = training_history[-1].get('test_accuracy', final_train_acc)
    total_params = training_history[0].get('total_params', 0) / 1000  # Assume in thousands
    
    metrics = ['Train Acc', 'Test Acc', 'Avg Time\n(seconds)']
    values = [final_train_acc, final_test_acc, np.mean(times)]
    
    bars = ax4.bar(metrics, values, color=['blue', 'green', 'orange'], alpha=0.7)
    ax4.set_title('Final Performance Summary')
    ax4.set_ylabel('Value')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_field_energy_evolution(model, text, device='cpu', save_path='field_energy_evolution.png'):
    """
    Plot how field energy evolves through layers
    """
    model.eval()
    
    byte_tensor = text_to_tensor(text, max_length=32, device=device)
    
    with torch.no_grad():
        positions = torch.linspace(0, 1, 64, device=device).unsqueeze(0)
        positions_np = positions.squeeze().cpu().numpy()
        
        # Track field through evolution
        current_field = model.local_field.compute_local_continuous_field(byte_tensor, positions)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        # Plot initial field energy
        energy = torch.sum(current_field**2, dim=-1).squeeze().cpu().numpy()
        axes[0].plot(positions_np, energy, 'b-', linewidth=2)
        axes[0].set_title('Initial Field Energy')
        axes[0].set_ylabel('Energy |Ψ(x)|²')
        axes[0].grid(True, alpha=0.3)
        
        # Plot evolution through layers
        for layer_idx, evolution_layer in enumerate(model.evolution_layers[:3]):
            evolved = evolution_layer(current_field)
            current_field = current_field + evolved
            
            energy = torch.sum(current_field**2, dim=-1).squeeze().cpu().numpy()
            axes[layer_idx + 1].plot(positions_np, energy, 'g-', linewidth=2)
            axes[layer_idx + 1].set_title(f'After Layer {layer_idx + 1}')
            axes[layer_idx + 1].set_ylabel('Energy |Ψ(x)|²')
            axes[layer_idx + 1].set_xlabel('Position x')
            axes[layer_idx + 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Field Energy Evolution: "{text}"', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_field_comparison(model, text_pairs, device='cpu', save_path='field_comparison.png'):
    """
    Compare semantic fields between different texts
    """
    model.eval()
    
    n_pairs = len(text_pairs)
    fig, axes = plt.subplots(n_pairs, 2, figsize=(12, 4 * n_pairs))
    
    if n_pairs == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for pair_idx, (text1, text2) in enumerate(text_pairs):
            for text_idx, text in enumerate([text1, text2]):
                byte_tensor = text_to_tensor(text, max_length=32, device=device)
                
                positions = torch.linspace(0, 1, 128, device=device).unsqueeze(0)
                positions_np = positions.squeeze().cpu().numpy()
                
                # Get final evolved field
                field = model.local_field.compute_local_continuous_field(byte_tensor, positions)
                
                for evolution_layer in model.evolution_layers:
                    evolved = evolution_layer(field)
                    field = field + evolved
                
                # Plot energy distribution
                energy = torch.sum(field**2, dim=-1).squeeze().cpu().numpy()
                
                axes[pair_idx, text_idx].plot(positions_np, energy, linewidth=2)
                axes[pair_idx, text_idx].fill_between(positions_np, energy, alpha=0.3)
                axes[pair_idx, text_idx].set_title(f'"{text[:30]}..."')
                axes[pair_idx, text_idx].set_ylabel('Energy |Ψ(x)|²')
                axes[pair_idx, text_idx].grid(True, alpha=0.3)
                
                if pair_idx == n_pairs - 1:
                    axes[pair_idx, text_idx].set_xlabel('Position x')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def create_visualization_suite(model, text_samples, training_history=None, device='cpu'):
    """
    Create a complete suite of visualizations
    """
    print("\n🎨 CREATING VISUALIZATION SUITE")
    print("-" * 60)
    
    # 1. Field dynamics
    plot_field_dynamics(model, text_samples, device, 'complete_field_dynamics.png')
    
    # 2. Training curves (if history provided)
    if training_history:
        plot_training_curves(training_history, 'complete_sft_performance.png')
    
    # 3. Field energy evolution for sample text
    if text_samples:
        plot_field_energy_evolution(model, text_samples[0], device, 'field_energy_evolution.png')
    
    # 4. Field comparison
    comparison_pairs = [
        ("The quick brown fox", "Brown fox the quick"),
        ("I love this movie", "I hate this movie")
    ]
    plot_field_comparison(model, comparison_pairs, device, 'field_comparison.png')
    
    print("✓ Visualization suite complete!")

def plot_similarity_heatmap(similarity_matrix, labels, title="Semantic Composition Similarity", save_path='semantic_composition.png'):
    """
    Plots a heatmap of the cosine similarity matrix.
    """
    print("\n🎨 VISUALIZING SEMANTIC SIMILARITY")
    print("-" * 60)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        similarity_matrix,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )
    ax.set_title(title, fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
