#!/usr/bin/env python3
"""
Core SFT v2 Field Components
Implements the advanced theoretical framework based on non-Abelian gauge theory.

- MultiComponentLocalField: Creates spinor-like semantic fields.
- GaugeFieldGenerator: Learns the syntactic gauge connection field.
- GaugeFieldEvolution: Implements field dynamics via the covariant derivative.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ============================================================================
# 1. SFT v2: SEMANTIC FIELD CONSTRUCTION (MULTI-COMPONENT)
# ============================================================================

class MultiComponentLocalField_v2(nn.Module):
    """
    Creates a multi-component (spinor-like) local semantic field.
    This field Ψ has components that can represent latent semantic roles.
    Output shape: [Batch, Position, N_Components, SemanticDim]
    """
    
    def __init__(self, semantic_dim=128, field_resolution=128, n_components=2):
        super().__init__()
        self.semantic_dim = semantic_dim
        self.field_resolution = field_resolution
        self.n_components = n_components
        
        # The embedding now generates a representation for all components
        self.byte_semantic_generators = nn.Embedding(256, semantic_dim * n_components)
        
        # Learnable locality parameters remain similar
        self.locality_shapes = nn.Parameter(torch.randn(256, 3))  # [offset, width, amplitude]

    def forward(self, byte_sequence, continuous_positions):
        """
        Create a continuous, multi-component field as a superposition of local contributions.
        """
        batch_size, seq_len = byte_sequence.shape
        _, n_positions = continuous_positions.shape
        
        # Get semantic amplitudes for each byte for all components
        byte_amplitudes_flat = self.byte_semantic_generators(byte_sequence)  # [B, L, D * Nc]
        byte_amplitudes = byte_amplitudes_flat.view(
            batch_size, seq_len, self.n_components, self.semantic_dim
        ) # [B, L, Nc, D]
        
        byte_positions = torch.linspace(0, 1, seq_len, device=byte_sequence.device)
        
        # Superposition of local contributions
        x_diff = continuous_positions.unsqueeze(-1) - byte_positions.unsqueeze(1) # [B, P, L]
        
        locality_params = self.locality_shapes[byte_sequence] # [B, L, 3]
        widths = torch.abs(locality_params[:, :, 1]) * 0.02 + 1e-5 # [B, L]
        amplitudes = torch.sigmoid(locality_params[:, :, 2]) # [B, L]
        
        # Gaussian basis functions for locality
        local_basis = torch.exp(-0.5 * (x_diff / widths.unsqueeze(1))**2) # [B, P, L]
        local_basis = local_basis * amplitudes.unsqueeze(1) # [B, P, L]
        
        # Contract basis with amplitudes to form the field
        # einsum: (batch, pos, len) * (batch, len, comps, dim) -> (batch, pos, comps, dim)
        semantic_field = torch.einsum('bpl,blcd->bpcd', local_basis, byte_amplitudes)
        
        return semantic_field

# ============================================================================
# 2. SFT v2: GAUGE FIELD GENERATOR (THE SYNTAX ENGINE)
# ============================================================================

class GaugeFieldGenerator_v2(nn.Module):
    """
    Generates the Gauge Connection Field A(x).
    This field represents the learned syntactic rules and applies role-binding.
    Output: A field of matrices, shape [Batch, Position, N_Components, N_Components]
    """
    def __init__(self, field_resolution=128, n_components=2):
        super().__init__()
        self.field_resolution = field_resolution
        self.n_components = n_components
        
        # A convolutional network to process the byte sequence and infer syntax
        self.syntax_encoder = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(128, 64, kernel_size=5, padding=2),
            nn.GELU(),
            # Output channels for each element of the Nc x Nc matrix
            nn.Conv1d(64, n_components * n_components, kernel_size=3, padding=1)
        )

    def forward(self, byte_sequence):
        """
        Generate the gauge field A(x) from the raw byte sequence.
        """
        batch_size, seq_len = byte_sequence.shape
        byte_onehot = F.one_hot(byte_sequence, 256).float().transpose(1, 2) # [B, 256, L]
        
        # Encode syntax from the sequence
        syntax_encoding = self.syntax_encoder(byte_onehot) # [B, Nc*Nc, L]
        
        # Upsample from sequence length to field resolution to create a continuous field
        gauge_field_flat = F.interpolate(
            syntax_encoding, 
            size=self.field_resolution, 
            mode='linear', 
            align_corners=False
        ) # [B, Nc*Nc, P]
        
        # Reshape into a field of matrices
        gauge_field = gauge_field_flat.permute(0, 2, 1).view(
            batch_size, self.field_resolution, self.n_components, self.n_components
        ) # [B, P, Nc, Nc]
        
        # Optional: Make the matrices skew-Hermitian (for unitary evolution)
        # This enforces certain conservation laws. A_dagger = -A
        gauge_field = (gauge_field - gauge_field.transpose(-1, -2).conj()) / 2.0

        return gauge_field

# ============================================================================
# 3. SFT v2: GAUGE FIELD EVOLUTION (THE NEW DYNAMICS)
# ============================================================================

class GaugeFieldEvolution_v2(nn.Module):
    """
    Evolves the semantic field using a Covariant Derivative.
    This combines spatial propagation with syntactic binding via the gauge field.
    Dynamics: i∂Ψ/∂t = DΨ, where D = (∂ - igA)
    """
    def __init__(self, semantic_dim=128, n_components=2, kinetic_strength=0.1):
        super().__init__()
        self.semantic_dim = semantic_dim
        self.n_components = n_components
        
        # Learnable coupling constant 'g' for the gauge interaction strength
        self.gauge_coupling_strength = nn.Parameter(torch.ones(1) * 0.5)
        self.kinetic_strength = nn.Parameter(torch.ones(1) * kinetic_strength)
        self.evolution_time = nn.Parameter(torch.ones(1) * 0.1)

    def compute_laplacian(self, psi):
        """ Computes ∇²Ψ on the multi-component field. """
        # psi shape: [B, P, Nc, D] (complex)
        psi_left = torch.roll(psi, shifts=1, dims=1)
        psi_right = torch.roll(psi, shifts=-1, dims=1)
        dx = 1.0 / psi.shape[1]
        laplacian = (psi_right - 2 * psi + psi_left) / (dx**2)
        return laplacian

    def forward(self, semantic_field, gauge_field):
        """
        Evolve the field using the covariant derivative.
        """
        # Convert semantic field to complex
        psi = torch.complex(semantic_field, torch.zeros_like(semantic_field))
        dt = self.evolution_time

        # 1. Kinetic Term (∂Ψ): Spatial propagation
        # This is our Laplacian, representing how fields spread and interfere locally.
        kinetic_term = -self.kinetic_strength * self.compute_laplacian(psi)
        
        # 2. Gauge Term (gAΨ): Syntactic binding
        # This is the core of the new theory. The gauge field 'A' "rotates"
        # the component spinors, effectively binding roles to the concepts.
        g = self.gauge_coupling_strength

        gauge_field = torch.complex(gauge_field, torch.zeros_like(gauge_field))
        # einsum: (b,p,i,j) * (b,p,j,d) -> (b,p,i,d)
        # where i,j are component indices
        gauge_term = -1j * g * torch.einsum('bpij,bpjd->bpid', gauge_field, psi)

        # 3. Covariant Derivative Action (DΨ = ∂Ψ - igAΨ)
        # We combine both effects. The evolution is driven by both local
        # spreading and syntactic role-binding transformations.
        covariant_derivative_action = kinetic_term + gauge_term
        
        # 4. Time Evolution Step
        evolved_psi = psi + dt * covariant_derivative_action
        
        # 5. Normalization (optional but good for stability)
        norm = torch.sqrt(torch.sum(torch.abs(evolved_psi)**2, dim=(-2, -1), keepdim=True))
        evolved_psi = evolved_psi / (norm + 1e-8)
        
        return evolved_psi.real


# --- For Reference: Old SFT v1 Components ---
# class LocalSemanticField(nn.Module): ...
# class InteractiveFieldEvolution(nn.Module): ...
# class SemanticFieldInterference(nn.Module): ...