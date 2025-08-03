#!/usr/bin/env python3
"""
Core field components for Semantic Field Transform
Modular components that can be easily imported and reused
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LocalSemanticField(nn.Module):
    """
    Creates truly LOCAL semantic fields where each byte contributes
    to a localized region, preserving word order and syntactic structure.
    
    FIXED: No global smearing - each position maintains local information
    """
    
    def __init__(self, semantic_dim=256, field_resolution=512, locality_width=0.02):
        super().__init__()
        self.semantic_dim = semantic_dim
        self.field_resolution = field_resolution
        self.locality_width = locality_width
        
        # Rich byte semantic representations
        self.byte_semantic_generators = nn.Embedding(256, semantic_dim)
        
        # Learnable locality parameters for each byte type
        self.locality_shapes = nn.Parameter(torch.randn(256, 4))  # [offset, width, amplitude, phase]
        
        # Context-aware amplitude modulation
        self.context_modulator = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(128, 64, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # Multi-scale locality (for capturing different linguistic structures)
        self.multiscale_widths = nn.Parameter(torch.tensor([0.01, 0.02, 0.05, 0.1]))
        self.scale_weights = nn.Parameter(torch.ones(4))
        
    def compute_local_continuous_field(self, byte_sequence, continuous_positions):
        """
        Create continuous field as superposition of LOCAL contributions
        Each byte creates a localized "wave packet" in semantic space
        """
        batch_size, seq_len = byte_sequence.shape
        _, n_positions = continuous_positions.shape
        
        # Get semantic amplitudes for each byte
        byte_amplitudes = self.byte_semantic_generators(byte_sequence)  # [B, L, D]
        
        # Compute context-dependent modulation
        byte_onehot = F.one_hot(byte_sequence, 256).float().transpose(1, 2)  # [B, 256, L]
        context_influence = self.context_modulator(byte_onehot).squeeze(1)  # [B, L]
        
        # Apply context modulation to amplitudes
        modulated_amplitudes = byte_amplitudes * context_influence.unsqueeze(-1)
        
        # Byte positions in continuous space [0,1]
        byte_positions = torch.linspace(0, 1, seq_len, device=byte_sequence.device)
        byte_positions = byte_positions.unsqueeze(0).repeat(batch_size, 1)  # [B, L]
        
        # Initialize semantic field
        semantic_field = torch.zeros(batch_size, n_positions, self.semantic_dim, 
                                   device=byte_sequence.device)
        
        # Create LOCAL field contributions for each byte
        for i in range(seq_len):
            current_byte = byte_sequence[:, i]  # [B]
            current_position = byte_positions[:, i]  # [B]
            current_amplitude = modulated_amplitudes[:, i]  # [B, D]
            
            # Get locality parameters for this byte type
            locality_params = self.locality_shapes[current_byte]  # [B, 4]
            
            # Multi-scale local basis functions
            total_contribution = torch.zeros(batch_size, n_positions, self.semantic_dim,
                                           device=byte_sequence.device)
            
            for scale_idx, base_width in enumerate(self.multiscale_widths):
                # Position offset and width for this scale
                position_offset = locality_params[:, 0] * 0.05  # Small learnable offset
                width = torch.abs(locality_params[:, 1]) * base_width + 1e-5
                amplitude_scale = torch.sigmoid(locality_params[:, 2])
                phase = locality_params[:, 3] * np.pi
                
                # Actual byte position with offset
                center_position = current_position + position_offset
                
                # Distance from center
                x_diff = continuous_positions - center_position.unsqueeze(1)  # [B, P]
                
                # Local basis function (Gaussian wave packet)
                local_basis = torch.exp(-0.5 * (x_diff / width.unsqueeze(1))**2)
                
                # Add phase modulation for richer representation
                phase_modulation = torch.cos(phase.unsqueeze(1) + 
                                           2 * np.pi * x_diff / width.unsqueeze(1))
                local_basis = local_basis * phase_modulation
                
                # Scale by amplitude and weight
                local_basis = local_basis * amplitude_scale.unsqueeze(1) * self.scale_weights[scale_idx]
                
                # Create field contribution
                contribution = torch.einsum('bp,bd->bpd', local_basis, current_amplitude)
                total_contribution += contribution * self.scale_weights[scale_idx]
            
            # Add to total field
            semantic_field += total_contribution
        
        return semantic_field


class InteractiveFieldEvolution(nn.Module):
    """
    TRUE field dynamics with spatial coupling via Laplacian
    Information propagates through the field creating genuine wave dynamics
    """
    
    def __init__(self, semantic_dim=256, kinetic_strength=0.1, potential_strength=0.05):
        super().__init__()
        self.semantic_dim = semantic_dim
        
        # Potential energy (local interactions)
        self.potential_hamiltonian = nn.Parameter(torch.randn(semantic_dim, semantic_dim) * potential_strength)
        
        # Kinetic energy coefficient (controls coupling strength)
        self.kinetic_coeff = nn.Parameter(torch.ones(1) * kinetic_strength)
        
        # Nonlinear potential for context-dependent evolution
        self.nonlinear_potential = nn.Sequential(
            nn.Linear(semantic_dim, semantic_dim // 2),
            nn.Tanh(),
            nn.Linear(semantic_dim // 2, semantic_dim)
        )
        
        # Learnable evolution time
        self.evolution_time = nn.Parameter(torch.ones(1) * 0.1)
        
    def compute_field_laplacian(self, psi):
        """
        Compute ∇²Ψ with proper boundary handling
        This creates the spatial coupling that enables true field dynamics
        """
        batch_size, n_positions, semantic_dim = psi.shape
        
        # Use circular boundary conditions for natural field dynamics
        psi_left = torch.roll(psi, shifts=1, dims=1)
        psi_right = torch.roll(psi, shifts=-1, dims=1)
        
        # Second-order finite difference: ∇²Ψ ≈ (Ψ(x+h) - 2Ψ(x) + Ψ(x-h))/h²
        dx = 1.0 / n_positions
        laplacian = (psi_right - 2 * psi + psi_left) / (dx**2)
        
        return laplacian
    
    def forward(self, semantic_field):
        """
        Evolve semantic field with true Schrödinger dynamics
        iℏ ∂Ψ/∂t = (-ℏ²/2m ∇² + V)Ψ
        """
        batch_size, n_positions, semantic_dim = semantic_field.shape
        
        # Convert to complex field for evolution
        psi = torch.complex(semantic_field, torch.zeros_like(semantic_field))
        
        # Make Hamiltonian Hermitian
        H_potential_real = (self.potential_hamiltonian + self.potential_hamiltonian.T) / 2
        H_potential = torch.complex(H_potential_real, torch.zeros_like(H_potential_real)) 
               
        dt = self.evolution_time
        
        # 1. KINETIC ENERGY: Spatial coupling via Laplacian
        laplacian_psi = self.compute_field_laplacian(psi)
        kinetic_term = -self.kinetic_coeff * laplacian_psi  # -ℏ²/2m ∇²Ψ
        
        # 2. LINEAR POTENTIAL: Standard Hamiltonian action
        linear_potential = torch.einsum('ij,bpj->bpi', H_potential, psi)
        
        # 3. NONLINEAR POTENTIAL: Context-dependent interactions
        field_magnitude = torch.abs(psi)
        nonlinear_term = self.nonlinear_potential(field_magnitude)
        nonlinear_potential = torch.complex(nonlinear_term, torch.zeros_like(nonlinear_term)) * psi
        
        # 4. TOTAL HAMILTONIAN ACTION
        H_psi = kinetic_term + linear_potential + 0.1 * nonlinear_potential
        
        # 5. TIME EVOLUTION (first-order approximation)
        evolved_psi = psi - 1j * dt * H_psi
        
        # 6. NORMALIZATION (preserve total semantic content)
        norm = torch.sqrt(torch.sum(torch.abs(evolved_psi)**2, dim=(1, 2), keepdim=True))
        evolved_psi = evolved_psi / (norm + 1e-8)
        
        return evolved_psi.real