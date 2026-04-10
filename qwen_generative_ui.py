import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# =============================================================================
# Configuration & Hyperparameters
# =============================================================================
class Config:
    # Qwen dimensions
    qwen_embed_dim = 4096
    
    # Layout Decoder Specs
    layout_vocab_size = 1100  # 1000 bins for coordinates + ~100 classes/special tokens
    layout_hidden_dim = 512
    layout_num_layers = 4
    layout_num_heads = 8
    max_layout_seq_len = 500
    
    # Structural Decoder Specs (Compact Custom Transformer for JSON)
    json_vocab_size = 2000  # Simplified vocab for JSON structure
    json_hidden_dim = 512
    json_num_layers = 4
    json_num_heads = 8
    max_json_seq_len = 1000
    
    # Special Tokens
    pad_token_id = 0
    sos_token_id = 1
    eos_token_id = 2

cfg = Config()

# =============================================================================
# Phase 1: Data Prep & Latent Space Loading
# =============================================================================
def load_qwen_embedding(filepath="embeddings.json", index=0):
    """Loads a pre-computed Qwen embedding vector from a JSON file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Error: The required file '{filepath}' was not found. Execution aborted.")
        
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # Strictly follow the provided JSON structure: List of dicts with an "embedding" key
    if not isinstance(data, list) or len(data) <= index:
        raise ValueError(f"Error: JSON data must be a list with at least {index + 1} elements.")
        
    element = data[index]
    if "embedding" not in element:
        raise KeyError(f"Error: 'embedding' key not found in the element at index {index}.")
        
    embedding_data = element["embedding"]
    
    tensor = torch.tensor(embedding_data, dtype=torch.float32).unsqueeze(0)
    return tensor

def mock_rico_data(batch_size=1):
    """Mocks extracted RICO data: Bounding Box sequences and JSON token sequences."""
    layout_seqs = torch.randint(3, cfg.layout_vocab_size, (batch_size, 50))
    json_seqs = torch.randint(3, cfg.json_vocab_size, (batch_size, 100))
    return layout_seqs, json_seqs

def load_json_element(filepath="data.json", index=0):
    """Loads a specific element from a JSON file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Error: The required file '{filepath}' was not found. Execution aborted.")
        
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, list) and len(data) > index:
            return data[index]
        return data

# =============================================================================
# Phase 2: Layout Decoder (Autoregressive Bounding Boxes)
# =============================================================================
class LayoutDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Project 4096d Qwen Vector -> Layout Hidden Dim (Conditioning)
        self.condition_proj = nn.Linear(config.qwen_embed_dim, config.layout_hidden_dim)
        
        self.embedding = nn.Embedding(config.layout_vocab_size, config.layout_hidden_dim)
        self.pos_encoder = nn.Embedding(config.max_layout_seq_len, config.layout_hidden_dim)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.layout_hidden_dim, 
            nhead=config.layout_num_heads, 
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, config.layout_num_layers)
        self.fc_out = nn.Linear(config.layout_hidden_dim, config.layout_vocab_size)
        
        self.config = config

    def forward(self, qwen_latent, tgt_seq, tgt_mask=None):
        """
        qwen_latent: (B, 4096)
        tgt_seq: (B, Seq_Len)
        """
        B, seq_len = tgt_seq.size()
        
        # Create Condition Prefix (B, 1, Hidden_Dim)
        condition = self.condition_proj(qwen_latent).unsqueeze(1)
        
        # Embed Target Sequence
        positions = torch.arange(0, seq_len, device=tgt_seq.device).unsqueeze(0).expand(B, seq_len)
        tgt_emb = self.embedding(tgt_seq) + self.pos_encoder(positions)
        
        # Transformer Decoder expects: tgt, memory
        # Here, memory is just our condition prefix.
        output = self.transformer_decoder(tgt_emb, condition, tgt_mask=tgt_mask)
        logits = self.fc_out(output)
        return logits

    def generate(self, qwen_latent, max_length=100):
        """Autoregressive generation for inference."""
        B = qwen_latent.size(0)
        device = qwen_latent.device
        
        # Start with SOS token
        current_seq = torch.full((B, 1), self.config.sos_token_id, dtype=torch.long, device=device)
        
        for _ in range(max_length):
            # Causal mask
            seq_len = current_seq.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
            
            logits = self(qwen_latent, current_seq, tgt_mask=tgt_mask)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            current_seq = torch.cat([current_seq, next_token], dim=1)
            
            # Simple stopping criteria (if all reached EOS, stop - omitted for simplicity)
            if (next_token == self.config.eos_token_id).all():
                break
                
        return current_seq

# =============================================================================
# Phase 3: Structural Decoder (JSON Tree)
# =============================================================================
class StructuralDecoder(nn.Module):
    def __init__(self, config):
        """
        A similar Transformer Decoder setup, but tuned for JSON code generation.
        In a full implementation, this could wrap a pretrained Qwen-Coder or LLaMA layer.
        """
        super().__init__()
        self.condition_proj = nn.Sequential(
            nn.Linear(config.qwen_embed_dim, config.json_hidden_dim * 2),
            nn.GELU(),
            nn.Linear(config.json_hidden_dim * 2, config.json_hidden_dim)
        )
        
        self.embedding = nn.Embedding(config.json_vocab_size, config.json_hidden_dim)
        self.pos_encoder = nn.Embedding(config.max_json_seq_len, config.json_hidden_dim)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.json_hidden_dim, 
            nhead=config.json_num_heads, 
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, config.json_num_layers)
        self.fc_out = nn.Linear(config.json_hidden_dim, config.json_vocab_size)
        self.config = config

    def forward(self, qwen_latent, tgt_seq, tgt_mask=None):
        B, seq_len = tgt_seq.size()
        condition = self.condition_proj(qwen_latent).unsqueeze(1)
        
        positions = torch.arange(0, seq_len, device=tgt_seq.device).unsqueeze(0).expand(B, seq_len)
        tgt_emb = self.embedding(tgt_seq) + self.pos_encoder(positions)
        
        output = self.transformer_decoder(tgt_emb, condition, tgt_mask=tgt_mask)
        logits = self.fc_out(output)
        return logits

    def generate(self, qwen_latent, max_length=200):
        # Similar autoregressive generation as Layout Decoder
        B = qwen_latent.size(0)
        device = qwen_latent.device
        current_seq = torch.full((B, 1), self.config.sos_token_id, dtype=torch.long, device=device)
        
        for _ in range(max_length):
            seq_len = current_seq.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
            logits = self(qwen_latent, current_seq, tgt_mask=tgt_mask)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            current_seq = torch.cat([current_seq, next_token], dim=1)
            if (next_token == self.config.eos_token_id).all():
                break
        return current_seq

# =============================================================================
# Phase 4: Inference Pipeline & Interpolation
# =============================================================================
def slerp(val, low, high):
    """Spherical Linear Interpolation for Latent Space Vectors."""
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    
    # Compute the cosine of the angle between the two vectors
    omega = torch.acos(torch.clamp((low_norm * high_norm).sum(1), -1, 1))
    so = torch.sin(omega)
    
    # If the angle is very small, fallback to linear interpolation
    if torch.all(so < 1e-5):
        return (1.0 - val) * low + val * high
        
    return torch.sin((1.0 - val) * omega) / so * low + torch.sin(val * omega) / so * high

# =============================================================================
# Phase 5: Token Translation & Decoding Functions
# =============================================================================
def decode_layout_tokens(token_list, config):
    """Translates raw token IDs into human-readable Bounding Boxes and UI Classes."""
    decoded_boxes = []
    # Skip Start-Of-Sequence token if present
    tokens = token_list[1:] if token_list and token_list[0] == config.sos_token_id else token_list
    
    # Simulate decoding: Group by 5 tokens [x_min, y_min, x_max, y_max, class_id]
    for i in range(0, len(tokens), 5):
        chunk = tokens[i:i+5]
        if config.eos_token_id in chunk:
            break
        if len(chunk) == 5:
            # Ensure proper min/max ordering for bounding box simulation
            x1, y1, x2, y2, cls_id = chunk
            box = {
                "class_id": f"UI_Element_{cls_id}",
                "bbox": [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            }
            decoded_boxes.append(box)
    return decoded_boxes

def decode_json_tokens(token_list, config):
    """Translates raw structural token IDs back into a valid JSON tree string."""
    # Skip Start-Of-Sequence token if present
    tokens = token_list[1:] if token_list and token_list[0] == config.sos_token_id else token_list
    
    elements = []
    for t in tokens:
        if t == config.eos_token_id:
            break
        if t not in [config.pad_token_id]:
            # Simulate mapping a token to a JSON node
            node_type = "container" if t % 2 == 0 else "component"
            elements.append({"node_id": t, "type": node_type})
            
    # Build a simulated valid DOM/JSON structure
    mock_tree = {
        "type": "root",
        "children": elements
    }
    return json.dumps(mock_tree, indent=2)

# =============================================================================
# Main Pipeline execution logic
# =============================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize Models
    layout_decoder = LayoutDecoder(cfg).to(device)
    structural_decoder = StructuralDecoder(cfg).to(device)
    
    # Mock Optimizers
    opt_layout = torch.optim.Adam(layout_decoder.parameters(), lr=1e-4)
    opt_struct = torch.optim.Adam(structural_decoder.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=cfg.pad_token_id)

    print("\n--- Starting Dry-Run Training Phase ---")
    
    batch_size = 2
    # Mock inputs for training dry-run (we simulate a batch of 2 embeddings here to ensure loss computes)
    latent_vecs_mock = torch.randn(batch_size, cfg.qwen_embed_dim).to(device)
    tgt_layout, tgt_json = mock_rico_data(batch_size)
    tgt_layout, tgt_json = tgt_layout.to(device), tgt_json.to(device)
    
    # Create causal masks
    layout_mask = nn.Transformer.generate_square_subsequent_mask(tgt_layout.size(1)).to(device)
    json_mask = nn.Transformer.generate_square_subsequent_mask(tgt_json.size(1)).to(device)

    # 1. Forward Pass: Layout
    opt_layout.zero_grad()
    layout_logits = layout_decoder(latent_vecs_mock, tgt_layout, tgt_mask=layout_mask)
    # Shift logits & targets for next-token prediction
    loss_layout = criterion(layout_logits[:, :-1, :].reshape(-1, cfg.layout_vocab_size), tgt_layout[:, 1:].reshape(-1))
    loss_layout.backward()
    opt_layout.step()
    print(f"Layout Forward/Backward successful. Mock Loss: {loss_layout.item():.4f}")

    # 2. Forward Pass: Structure
    opt_struct.zero_grad()
    json_logits = structural_decoder(latent_vecs_mock, tgt_json, tgt_mask=json_mask)
    loss_struct = criterion(json_logits[:, :-1, :].reshape(-1, cfg.json_vocab_size), tgt_json[:, 1:].reshape(-1))
    loss_struct.backward()
    opt_struct.step()
    print(f"Structural Forward/Backward successful. Mock Loss: {loss_struct.item():.4f}")

    print("\n--- Starting Phase 4: Inference Pipeline (Real Embeddings Load) ---")
    
    layout_decoder.eval()
    structural_decoder.eval()
    
    with torch.no_grad():
        # Load the real Qwen embedding from embeddings.json
        print("Loading actual Qwen embedding from 'embeddings.json'...")
        try:
            vec_input = load_qwen_embedding("embeddings.json", index=0).to(device)
            print(f"Successfully loaded true embedding of shape: {vec_input.shape}")
        except Exception as e:
            print(f"Critical error loading embeddings: {e}")
            return
            
        print("Generating UI layout and structure for the loaded embedding...")
        gen_layout = layout_decoder.generate(vec_input, max_length=20)
        gen_json = structural_decoder.generate(vec_input, max_length=30)
        
        layout_tokens_list = gen_layout[0].tolist()
        json_tokens_list = gen_json[0].tolist()
        
        print("\n--- Starting Phase 5: Decoding Tokens ---")
        
        print(f"Raw Layout Tokens (Shape: {gen_layout.shape}):")
        print(layout_tokens_list)
        print("\nDecoded Layout (Valid Bounding Boxes):")
        decoded_layout = decode_layout_tokens(layout_tokens_list, cfg)
        print(json.dumps(decoded_layout, indent=2))
        
        print(f"\nRaw JSON Tokens (Shape: {gen_json.shape}):")
        print(json_tokens_list)
        print("\nDecoded JSON (Valid Code Structure):")
        decoded_json = decode_json_tokens(json_tokens_list, cfg)
        print(decoded_json)
        
        print("\nUI Generation from Qwen Embedding completed!")

if __name__ == "__main__":
    main()
