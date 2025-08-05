#!/usr/bin/env python3
"""
Core SFT v2.1 Field Components - Position-Aware Version
The definitive implementation that correctly provides the model with an
explicit coordinate system, enabling the learning of position-dependent syntax.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ============================================================================
# 1. POSITIONAL ENCODING MODULE
# ============================================================================

class PositionalEncoding(nn.Module):
    """ Standard sinusoidal positional encoding. """
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [Batch, SeqLen, EmbeddingDim]
        return x + self.pe[:, :x.size(1), :]

# ============================================================================
# 2. SFT v2.1: POSITION-AWARE FIELD CONSTRUCTION
# ============================================================================

class MultiComponentLocalField_v2(nn.Module):
    def __init__(self, semantic_dim=128, n_components=2, field_resolution=128):
        super().__init__()
        self.semantic_dim = semantic_dim
        self.n_components = n_components
        self.embedding_dim = semantic_dim * n_components
        self.field_resolution = field_resolution

        self.byte_embedder = nn.Embedding(256, self.embedding_dim)
        self.pos_encoder = PositionalEncoding(self.embedding_dim)
        self.locality_shapes = nn.Parameter(torch.randn(256, 3))

    def forward(self, byte_sequence, continuous_positions):
        batch_size, seq_len = byte_sequence.shape
        
        byte_embeds = self.byte_embedder(byte_sequence)
        pos_aware_embeds = self.pos_encoder(byte_embeds)
        
        # --- POTENTIAL BUG IS HERE ---
        # Original: .view(batch_size, seq_len, self.n_components, self.semantic_dim) -> [B, L, 2, 64]
        # Let's ensure this order is correct. This should be right.
        byte_amplitudes = pos_aware_embeds.view(
            batch_size, seq_len, self.n_components, self.semantic_dim
        )
        
        byte_positions = torch.linspace(0, 1, seq_len, device=byte_sequence.device)
        x_diff = continuous_positions.unsqueeze(-1) - byte_positions.unsqueeze(1)
        
        locality_params = self.locality_shapes[byte_sequence]
        widths = torch.abs(locality_params[:, :, 1]) * 0.02 + 1e-5
        amplitudes = torch.sigmoid(locality_params[:, :, 2])
        
        local_basis = torch.exp(-0.5 * (x_diff / widths.unsqueeze(1))**2)
        local_basis = local_basis * amplitudes.unsqueeze(1)
        
        # einsum: (b,p,l) * (b,l,c,d) -> (b,p,c,d)
        semantic_field = torch.einsum('bpl,blcd->bpcd', local_basis, byte_amplitudes)
        
        return semantic_field

# ============================================================================
# 3. SFT v2.1: POSITION-AWARE GAUGE FIELD GENERATOR
# ============================================================================

class GaugeFieldGenerator_v2(nn.Module):
    """
    Generates the Gauge Connection Field A(x).
    CRITICAL UPGRADE v2.1: The syntax encoder now operates on the EVOLVED
    field from the previous layer, allowing syntax to be inferred from semantics.
    """
    def __init__(self, field_resolution=128, n_components=2, input_dim=64): # input_dim is now semantic_dim
        super().__init__()
        self.field_resolution = field_resolution
        self.n_components = n_components
        
        # The input is now the flattened semantic field [B, Nc*D, P]
        input_channels = n_components * input_dim

        # A convolutional network to process the semantic field and infer syntax
        self.syntax_encoder = nn.Sequential(
            nn.Conv1d(input_channels, 128, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(64, n_components * n_components, kernel_size=1)
        )

    def forward(self, semantic_field):
        """
        Generate the gauge field A(x) from the current semantic field state.
        """
        batch_size, resolution, n_c, n_d = semantic_field.shape
        
        # Reshape the field to be [B, C, L] for the Conv1d
        # [B, P, Nc, D] -> [B, P, Nc*D] -> [B, Nc*D, P]
        field_for_conv = semantic_field.view(batch_size, resolution, -1).transpose(1, 2)
        
        # Encode syntax from the current field state
        syntax_encoding = self.syntax_encoder(field_for_conv) # [B, Nc*Nc, P]
        
        # Reshape into a field of matrices
        gauge_field = syntax_encoding.permute(0, 2, 1).view(
            batch_size, self.field_resolution, self.n_components, self.n_components
        )
        
        # Make the matrices skew-Hermitian for unitary evolution
        gauge_field = (gauge_field - gauge_field.transpose(-1, -2).conj()) / 2.0

        return gauge_field

# ============================================================================
# 4. SFT v2.1: FIELD EVOLUTION (UNCHANGED)
# ============================================================================
# The evolution engine remains the same. The change is in the fields it receives.
class GaugeFieldEvolution_v2(nn.Module):
    def __init__(self, semantic_dim=128, n_components=2, kinetic_strength=0.1):
        super().__init__()
        self.semantic_dim = semantic_dim
        self.n_components = n_components
        self.gauge_coupling_strength = nn.Parameter(torch.ones(1) * 0.5)
        self.kinetic_strength = nn.Parameter(torch.ones(1) * kinetic_strength)
        self.evolution_time = nn.Parameter(torch.ones(1) * 0.1)

    def compute_laplacian(self, psi):
        psi_left = torch.roll(psi, shifts=1, dims=1)
        psi_right = torch.roll(psi, shifts=-1, dims=1)
        dx = 1.0 / psi.shape[1]
        laplacian = (psi_right - 2 * psi + psi_left) / (dx**2)
        return laplacian

    def forward(self, semantic_field, gauge_field):
        psi = torch.complex(semantic_field, torch.zeros_like(semantic_field))
        dt = self.evolution_time
        kinetic_term = -self.kinetic_strength * self.compute_laplacian(psi)
        g = self.gauge_coupling_strength
        
        # --- ROBUST MATMUL FIX ---
        # psi shape is [B, P, Nc, D]
        # gauge_field shape is [B, P, Nc, Nc]
        
        # We want to perform a matmul for each P: gauge[Nc,Nc] @ psi[Nc,D]
        # torch.matmul handles batch dimensions (B and P) automatically
        gauge_field = torch.complex(gauge_field, torch.zeros_like(gauge_field))
        gauge_effect = torch.matmul(gauge_field, psi)
        
        gauge_term = -1j * g * gauge_effect
        # --- END OF FIX ---

        covariant_derivative_action = kinetic_term + gauge_term
        evolved_psi = psi + dt * covariant_derivative_action
        norm = torch.sqrt(torch.sum(torch.abs(evolved_psi)**2, dim=(-2, -1), keepdim=True))
        evolved_psi = evolved_psi / (norm + 1e-8)
        return evolved_psi.real