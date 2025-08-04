#!/usr/bin/env python3
"""
SFT v2 Experiment Runner
Orchestrates the crucial SVO composition experiment for the new architecture.
"""

import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.sft_model import CompleteGenuineSFT_v2
from utils.data_utils import create_svo_compositional_dataset, create_data_loaders
from utils.training_utils import SFTTrainer
from utils.validation_utils import run_svo_composition_test
from utils.visualization_utils import plot_similarity_heatmap

class SFT_v2_Experiment:
    """
    Orchestrates the SVO experiment for the SFT v2 model.
    """
    
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE
        print(f"Using device: {self.device}")
        
        self.model = CompleteGenuineSFT_v2(
            semantic_dim=config.SEMANTIC_DIM,
            field_resolution=config.FIELD_RESOLUTION,
            n_layers=config.N_LAYERS,
            n_components=2, # Hard-coded for SU(2)-like theory
            n_classes=config.N_CLASSES
        ).to(self.device)
        
    def run(self):
        """
        Run the complete SVO experiment pipeline.
        """
        print("\n🔬 RUNNING SVO COMPOSITION EXPERIMENT FOR SFT v2")
        print("=" * 80)
        
        # 1. Create the specialized SVO dataset
        svo_dataset = create_svo_compositional_dataset()
        train_data = svo_dataset['train']
        
        train_loader, _ = create_data_loaders(
            train_data['texts'], train_data['labels'],
            batch_size=self.config.BATCH_SIZE, test_split=0.01
        )
        
        # 2. Train the model on the non-compositional data
        print("\n🚀 TRAINING SFT v2 ON AGENT vs. ACTION/OBJECT DATA")
        trainer = SFTTrainer(self.model, device=self.device, lr=self.config.LEARNING_RATE)
        trainer.train(train_loader, epochs=self.config.EPOCHS, verbose=True)
        print("Training complete.")

        # 3. Run the semantic role validation test
        similarity_matrix, labels = run_svo_composition_test(
            self.model, svo_dataset, device=self.device
        )
        
        # 4. Visualize the results
        plot_similarity_heatmap(
            similarity_matrix, 
            labels, 
            title="SFT v2: SVO Semantic Role Similarity",
            save_path="svo_composition_v2.png"
        )
        
        print("\n✅ SFT v2 SVO Experiment Finished.")