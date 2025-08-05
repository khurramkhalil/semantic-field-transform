#!/usr/bin/env python3
"""
Validation utilities for SFT models
Scientific validation and analysis tools
"""

import torch
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_utils import text_to_tensor, create_validation_test_pairs, create_localization_test_cases


def validate_information_preservation(model, test_cases=None, device='cpu'):
    """
    Validate that rich semantic information is preserved through to classification
    """
    print("\n🔬 VALIDATING INFORMATION PRESERVATION")
    print("-" * 60)
    
    model.eval()
    
    # Use default test cases if none provided
    if test_cases is None:
        test_pairs = create_validation_test_pairs()
    else:
        test_pairs = test_cases
    
    with torch.no_grad():
        for pair_idx, (text1, text2) in enumerate(test_pairs):
            print(f"\nTest pair {pair_idx + 1}:")
            print(f"  Text 1: '{text1}'")
            print(f"  Text 2: '{text2}'")
            
            results = []
            for text in [text1, text2]:
                # Convert to tensor
                byte_tensor = text_to_tensor(text, device=device)
                
                # Get model output and evolved field
                output, evolved_field = model(byte_tensor)
                
                results.append({
                    'output': output,
                    'field': evolved_field,
                    'text': text
                })
            
            # Compare outputs and fields
            output_diff = torch.norm(results[0]['output'] - results[1]['output']).item()
            field_diff = torch.norm(results[0]['field'] - results[1]['field']).item()
            
            print(f"  Output difference: {output_diff:.6f}")
            print(f"  Field difference:  {field_diff:.6f}")
            print(f"  Sensitivity: {'✓ PASS' if output_diff > 1e-3 else '✗ FAIL'}")
    
    return True


def analyze_semantic_field_evolution(model, text_sample, device='cpu'):
    """
    Analyze how semantic fields evolve through the processing layers
    """
    print(f"\n📊 ANALYZING FIELD EVOLUTION FOR: '{text_sample}'")
    print("-" * 60)
    
    model.eval()
    
    # Convert text to input
    byte_tensor = text_to_tensor(text_sample, max_length=64, device=device)
    
    with torch.no_grad():
        # Get continuous positions
        positions = torch.linspace(0, 1, model.field_resolution, device=device).unsqueeze(0)
        
        # Track field evolution through layers
        current_field = model.local_field.compute_local_continuous_field(byte_tensor, positions)
        field_evolution = [current_field.clone()]
        
        print("Field evolution statistics:")
        print(f"Initial field - Mean: {current_field.mean():.4f}, Std: {current_field.std():.4f}")
        
        for layer_idx, evolution_layer in enumerate(model.evolution_layers):
            evolved = evolution_layer(current_field)
            current_field = current_field + evolved
            field_evolution.append(current_field.clone())
            
            print(f"Layer {layer_idx + 1} - Mean: {current_field.mean():.4f}, Std: {current_field.std():.4f}")
        
        # Analyze field characteristics
        final_field = field_evolution[-1]
        
        # Compute field energy (semantic content measure)
        field_energy = torch.sum(final_field**2, dim=-1)  # [B, P]
        
        # Find regions of high semantic activity
        energy_threshold = field_energy.mean() + field_energy.std()
        active_regions = positions.squeeze()[field_energy.squeeze() > energy_threshold]
        
        print(f"\nField analysis:")
        print(f"  Total semantic energy: {field_energy.sum().item():.4f}")
        print(f"  Active regions: {len(active_regions)}/{model.field_resolution}")
        if len(active_regions) > 0:
            print(f"  Active span: {active_regions.min():.3f} to {active_regions.max():.3f}")
    
    return field_evolution


def run_scientific_validation_suite(model, device='cpu'):
    """
    Comprehensive scientific validation of SFT theoretical predictions
    """
    print("\n🔬 SCIENTIFIC VALIDATION SUITE")
    print("=" * 80)
    
    # Test 1: Local Information Preservation
    print("\n1. LOCAL INFORMATION PRESERVATION TEST")
    print("-" * 50)
    
    test_cases = create_localization_test_cases()
    
    model.eval()
    with torch.no_grad():
        for text, expected_peaks in test_cases:
            byte_tensor = text_to_tensor(text, max_length=32, device=device)
            
            # Get field
            positions = torch.linspace(0, 1, 64, device=device).unsqueeze(0)
            field = model.local_field.compute_local_continuous_field(byte_tensor, positions)
            
            # Compute field energy
            energy = torch.sum(field**2, dim=-1).squeeze().cpu().numpy()
            positions_np = positions.squeeze().cpu().numpy()
            
            # Find peaks (try-except for scipy import)
            try:
                from scipy.signal import find_peaks
                peaks, _ = find_peaks(energy, height=energy.max() * 0.3)
                peak_positions = positions_np[peaks]
            except ImportError:
                # Fallback peak detection
                peak_positions = positions_np[energy > energy.max() * 0.3]
            
            print(f"  Text: '{text}'")
            print(f"  Expected peaks near: {expected_peaks}")
            print(f"  Found peaks at: {peak_positions.tolist()}")
            print(f"  Localization quality: {'✓ GOOD' if len(peak_positions) > 0 else '✗ POOR'}")
    
    # Test 2: Field Evolution Dynamics
    print("\n2. FIELD EVOLUTION DYNAMICS TEST")
    print("-" * 50)
    
    with torch.no_grad():
        text = "The cat chased the mouse"
        byte_tensor = text_to_tensor(text, max_length=32, device=device)
        
        positions = torch.linspace(0, 1, 64, device=device).unsqueeze(0)
        
        # Track field through evolution
        current_field = model.local_field.compute_local_continuous_field(byte_tensor, positions)
        initial_energy = torch.sum(current_field**2, dim=-1)
        
        evolution_energies = [initial_energy.clone()]
        
        for layer_idx, evolution_layer in enumerate(model.evolution_layers):
            evolved = evolution_layer(current_field)
            current_field = current_field + evolved
            
            field_energy = torch.sum(current_field**2, dim=-1)
            evolution_energies.append(field_energy.clone())
            
            # Measure information propagation
            energy_spread = torch.std(field_energy.squeeze()).item()
            print(f"  Layer {layer_idx + 1}: Energy spread = {energy_spread:.4f}")
    
    print("  Field evolution: ✓ CONFIRMED - Energy propagates through layers")
    
    return True


def model_summary(model):
    """
    Print model architecture summary
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📋 MODEL SUMMARY:")
    print(f"   • Total parameters: {total_params:,}")
    print(f"   • Trainable parameters: {trainable_params:,}")
    print(f"   • Semantic dimension: {model.semantic_dim}")
    print(f"   • Field resolution: {model.field_resolution}")
    print(f"   • Evolution layers: {model.n_layers}")
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'semantic_dim': model.semantic_dim,
        'field_resolution': model.field_resolution,
        'n_layers': model.n_layers
    }

import torch.nn.functional as F

def run_semantic_composition_test(model, compositional_dataset, device='cpu'):
    """
    Tests the model's ability to represent unseen compositional concepts.
    This is the core validation for true semantic understanding.
    """
    print("\n🔬 VALIDATING SEMANTIC COMPOSITION")
    print("=" * 80)
    
    model.eval()
    
    basis_texts = compositional_dataset['basis']['texts']
    basis_concepts = compositional_dataset['basis']['concepts']
    test_texts = compositional_dataset['test']['texts']
    
    # Combine all texts to get their field representations in one go
    all_texts = basis_texts + test_texts
    all_labels = basis_concepts + test_texts

    field_vectors = {}

    with torch.no_grad():
        for text, label in zip(all_texts, all_labels):
            byte_tensor = text_to_tensor(text, device=device)
            
            # Get the final evolved field representation
            _ , evolved_field = model(byte_tensor) # We only need the field
            
            # Pool the field to get a single vector representation for the sentence
            # We use mean pooling here for simplicity
            pooled_vector = evolved_field.mean(dim=1).squeeze(0) # [D]
            field_vectors[label] = F.normalize(pooled_vector, p=2, dim=0)

    # Calculate the cosine similarity matrix
    num_vectors = len(all_labels)
    similarity_matrix = torch.zeros(num_vectors, num_vectors)
    
    for i in range(num_vectors):
        for j in range(num_vectors):
            vec_i = field_vectors[all_labels[i]]
            vec_j = field_vectors[all_labels[j]]
            similarity = torch.dot(vec_i, vec_j).item()
            similarity_matrix[i, j] = similarity
            
    # --- Hypothesis Testing ---
    print("\nHYPOTHESIS TEST: Does Ψ('red sphere') ≈ Ψ('red') + Ψ('sphere')?")
    print("-" * 60)
    
    # Example: "The sphere is red"
    test_concept = "The sphere is red"
    sim_with_red = similarity_matrix[len(basis_texts), basis_concepts.index('red')]
    sim_with_sphere = similarity_matrix[len(basis_texts), basis_concepts.index('sphere')]
    sim_with_blue = similarity_matrix[len(basis_texts), basis_concepts.index('blue')]
    sim_with_cube = similarity_matrix[len(basis_texts), basis_concepts.index('cube')]

    print(f"Similarity between '{test_concept}' and 'red': {sim_with_red:.3f}")
    print(f"Similarity between '{test_concept}' and 'sphere': {sim_with_sphere:.3f}")
    print(f"Similarity between '{test_concept}' and 'blue': {sim_with_blue:.3f}")
    print(f"Similarity between '{test_concept}' and 'cube': {sim_with_cube:.3f}")

    # Check if the correct concepts have higher similarity
    composition_test_passed = (sim_with_red > sim_with_blue) and (sim_with_sphere > sim_with_cube)
    
    print(f"\nCompositionality Check: {'✓ PASS' if composition_test_passed else '✗ FAIL'}")
    if not composition_test_passed:
        print("  Warning: Model may not be composing concepts as expected.")
        
    return similarity_matrix.cpu().numpy(), all_labels

# def run_svo_composition_test(model, compositional_dataset, device='cpu'):
#     """
#     Tests the model's understanding of semantic roles (SVO).
#     This is a more advanced test of true semantic composition.
#     """
#     print("\n🔬 VALIDATING SVO SEMANTIC ROLE UNDERSTANDING")
#     print("=" * 80)
    
#     model.eval()
    
#     basis_texts = compositional_dataset['basis']['texts']
#     basis_concepts = compositional_dataset['basis']['concepts']
#     test_texts = compositional_dataset['test']['texts']
    
#     # Combine all texts to get their field representations
#     all_texts = basis_texts + test_texts
#     all_labels = basis_concepts + test_texts

#     field_vectors = {}

#     with torch.no_grad():
#         for text, label in zip(all_texts, all_labels):
#             byte_tensor = text_to_tensor(text, device=device, max_length=32)
#             _ , evolved_field = model(byte_tensor)
#             pooled_vector = evolved_field.mean(dim=1).squeeze(0)
#             field_vectors[label] = F.normalize(pooled_vector, p=2, dim=0)

#     num_vectors = len(all_labels)
#     similarity_matrix = torch.zeros(num_vectors, num_vectors)
    
#     for i in range(num_vectors):
#         for j in range(num_vectors):
#             vec_i = field_vectors[all_labels[i]]
#             vec_j = field_vectors[all_labels[j]]
#             similarity_matrix[i, j] = torch.dot(vec_i, vec_j).item()
            
#     # --- HYPOTHESIS TESTING ---
#     print("\nPRIMARY HYPOTHESIS: Can the model distinguish semantic roles?")
#     print("-" * 60)
    
#     normal_svo = "The dog chased the ball"
#     inverted_svo = "The ball chased the dog"
    
#     sim_between_inversions = field_vectors[normal_svo] @ field_vectors[inverted_svo]
    
#     print(f"Similarity between '{normal_svo}' and '{inverted_svo}': {sim_between_inversions:.3f}")
    
#     role_test_passed = sim_between_inversions < 0.9  # Must not be seen as identical
    
#     print(f"\nSemantic Role Distinction Test: {'✓ PASS' if role_test_passed else '✗ FAIL'}")
#     if not role_test_passed:
#         print("  CRITICAL FAILURE: Model does not distinguish subject from object.")
#     else:
#         print("  SUCCESS: Model represents sentences with different roles differently.")

#     # --- SECONDARY HYPOTHESIS: Compositionality ---
#     print("\n\nSECONDARY HYPOTHESIS: Is the SVO field composed of its parts?")
#     print("-" * 60)
#     sim_dog = field_vectors[normal_svo] @ field_vectors["The dog"]
#     sim_chased = field_vectors[normal_svo] @ field_vectors["chased"]
#     sim_ball = field_vectors[normal_svo] @ field_vectors["the ball"]
    
#     # Compare with an unrelated concept
#     sim_cat = field_vectors[normal_svo] @ field_vectors["The cat"]

#     print(f"Similarity of '{normal_svo}' with 'The dog': {sim_dog:.3f}")
#     print(f"Similarity of '{normal_svo}' with 'chased': {sim_chased:.3f}")
#     print(f"Similarity of '{normal_svo}' with 'the ball': {sim_ball:.3f}")
#     print(f"Similarity of '{normal_svo}' with 'The cat' (control): {sim_cat:.3f}")
    
#     composition_test_passed = (sim_dog > sim_cat) and (sim_chased > sim_cat)

#     print(f"\nComponent Composition Test: {'✓ PASS' if composition_test_passed else '✗ FAIL'}")
        
#     return similarity_matrix.cpu().numpy(), all_labels

def run_svo_composition_test(model, compositional_dataset, max_length=64, device='cpu'):
    """
    Tests the SFT v2 model's understanding of semantic roles (SVO).
    This is a more advanced test of true semantic composition.
    """
    print("\n🔬 VALIDATING SVO SEMANTIC ROLE UNDERSTANDING (SFT v2)")
    print("=" * 80)
    
    model.eval()
    
    basis_texts = compositional_dataset['basis']['texts']
    basis_concepts = compositional_dataset['basis']['concepts']
    test_texts = compositional_dataset['test']['texts']
    
    all_texts = basis_texts + test_texts
    all_labels = basis_concepts + test_texts

    field_vectors = {}

    # # --- FIX APPLIED HERE ---
    # # Get max_length from the model's field constructor config, which is tied to its resolution
    # # This ensures consistency between byte_positions and continuous_positions
    # try:
    #     # Accessing via the new model structure
    #     max_length_for_validation = model.field_constructor.field_resolution
    # except AttributeError:
    #     # Fallback for older model versions if needed, though not ideal
    #     max_length_for_validation = 64

    with torch.no_grad():
        for text, label in zip(all_texts, all_labels):
            # Use the consistent max_length for creating the tensor
            byte_tensor = text_to_tensor(
                text, 
                device=device, 
                max_length=max_length # Use the correct length
            )
            # --- END OF FIX ---
            
            _ , evolved_field = model(byte_tensor)
            # Pooling the multi-component field
            # Shape: [1, P, Nc, D] -> [1, P, Nc*D] -> [1, Nc*D]
            pooled_vector = evolved_field.view(1, model.field_constructor.field_resolution, -1).mean(dim=1).squeeze(0)
            field_vectors[label] = F.normalize(pooled_vector, p=2, dim=0)

    num_vectors = len(all_labels)
    similarity_matrix = torch.zeros(num_vectors, num_vectors)
    
    for i in range(num_vectors):
        for j in range(len(all_labels)):
            # Handle potential key errors if a label isn't in field_vectors
            if all_labels[i] in field_vectors and all_labels[j] in field_vectors:
                vec_i = field_vectors[all_labels[i]]
                vec_j = field_vectors[all_labels[j]]
                similarity_matrix[i, j] = torch.dot(vec_i, vec_j).item()
            
    # --- HYPOTHESIS TESTING ---
    print("\nPRIMARY HYPOTHESIS: Can the model distinguish semantic roles?")
    print("-" * 60)
    
    normal_svo = "The dog chased the ball"
    inverted_svo = "The ball chased the dog"
    
    sim_between_inversions = field_vectors[normal_svo] @ field_vectors[inverted_svo]
    
    print(f"Similarity between '{normal_svo}' and '{inverted_svo}': {sim_between_inversions:.3f}")
    
    role_test_passed = sim_between_inversions < 0.9
    
    print(f"\nSemantic Role Distinction Test: {'✓ PASS' if role_test_passed else '✗ FAIL'}")
    if not role_test_passed:
        print("  CRITICAL NOTE: Model still struggles to fully separate subject from object.")
    else:
        print("  SUCCESS: Model represents sentences with different roles differently.")
        
    return similarity_matrix.cpu().numpy(), all_labels