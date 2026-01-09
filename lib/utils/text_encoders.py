"""
Text encoders for handling long text sequences.

Supports multiple text encoders:
1. CLIP (default, ~77 tokens max)
2. Sentence-BERT (all-mpnet-base-v2, ~512 tokens, recommended for long text)
3. Sentence-BERT (all-MiniLM-L6-v2, ~512 tokens, faster, smaller)
4. T5 Encoder (t5-base, ~512 tokens, good for understanding)
5. Longformer (longformer-base-4096, ~4096 tokens, for very long text)
"""

import torch
import torch.nn as nn
from typing import List, Optional


class CLIPTextEncoder(nn.Module):
    """Original CLIP text encoder (77 tokens max)."""
    def __init__(self, model_name="ViT-L-14", pretrained="openai", device="cuda"):
        super().__init__()
        import open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.embedding_dim = 768
        self.max_length = 77  # CLIP's limitation
        self.device = device
        self.model = self.model.to(device)
        for p in self.model.parameters():
            p.requires_grad = False
    
    @torch.no_grad()
    def encode(self, text_list: List[str]) -> torch.Tensor:
        """Encode text to embeddings."""
        # CLIP tokenizer handles lowercasing and truncation
        text_tokens = self.tokenizer([text.strip().lower() for text in text_list]).to(self.device)
        text_embeds = self.model.encode_text(text_tokens)
        # Normalize embeddings (critical for CLIP alignment)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        return text_embeds


class SentenceBERTEncoder(nn.Module):
    """Sentence-BERT encoder for longer text sequences (~512 tokens).
    
    Options:
    - 'all-mpnet-base-v2': 768 dim, best quality
    - 'all-MiniLM-L6-v2': 384 dim, faster, smaller
    """
    def __init__(self, model_name="all-mpnet-base-v2", device="cuda"):
        super().__init__()
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.max_length = 512  # Sentence-BERT typical max length
        self.device = device
        # Freeze encoder by default
        for p in self.model.parameters():
            p.requires_grad = False
    
    @torch.no_grad()
    def encode(self, text_list: List[str]) -> torch.Tensor:
        """Encode text to embeddings."""
        # SentenceTransformer handles tokenization and encoding
        text_embeds = self.model.encode(
            text_list,
            convert_to_tensor=True,
            device=self.device,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        return text_embeds


class T5Encoder(nn.Module):
    """T5 encoder-only model for longer text sequences (~512 tokens).
    
    Good for understanding long text sequences.
    Options: 't5-small' (512 dim), 't5-base' (768 dim)
    """
    def __init__(self, model_name="t5-base", device="cuda"):
        super().__init__()
        from transformers import T5EncoderModel, T5Tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5EncoderModel.from_pretrained(model_name).to(device)
        self.embedding_dim = self.model.config.d_model  # 768 for t5-base, 512 for t5-small
        self.max_length = 512
        self.device = device
        # Freeze encoder by default
        for p in self.model.parameters():
            p.requires_grad = False
    
    @torch.no_grad()
    def encode(self, text_list: List[str]) -> torch.Tensor:
        """Encode text to embeddings."""
        # Tokenize and truncate
        encoded = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)
        
        # Get encoder outputs
        outputs = self.model(**encoded)
        
        # Use mean pooling over sequence length to get fixed-size embeddings
        # outputs.last_hidden_state: [batch_size, seq_len, hidden_dim]
        attention_mask = encoded['attention_mask']
        embeddings = outputs.last_hidden_state
        # Masked mean pooling
        mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        text_embeds = sum_embeddings / sum_mask
        
        # Normalize embeddings
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        return text_embeds


class LongformerEncoder(nn.Module):
    """Longformer encoder for very long text sequences (~4096 tokens).
    
    Specifically designed for long documents.
    Options: 'allenai/longformer-base-4096'
    """
    def __init__(self, model_name="allenai/longformer-base-4096", device="cuda"):
        super().__init__()
        from transformers import LongformerModel, LongformerTokenizer
        self.tokenizer = LongformerTokenizer.from_pretrained(model_name)
        self.model = LongformerModel.from_pretrained(model_name).to(device)
        self.embedding_dim = self.model.config.hidden_size  # 768
        self.max_length = 4096
        self.device = device
        # Freeze encoder by default
        for p in self.model.parameters():
            p.requires_grad = False
    
    @torch.no_grad()
    def encode(self, text_list: List[str]) -> torch.Tensor:
        """Encode text to embeddings."""
        # Tokenize and truncate
        encoded = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)
        
        # Get encoder outputs
        outputs = self.model(**encoded)
        
        # Use [CLS] token or mean pooling
        # Longformer uses global attention on [CLS] token
        text_embeds = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        # Alternative: mean pooling
        # attention_mask = encoded['attention_mask']
        # embeddings = outputs.last_hidden_state
        # mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        # sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
        # sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        # text_embeds = sum_embeddings / sum_mask
        
        # Normalize embeddings
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        return text_embeds


def create_text_encoder(encoder_type: str = "sentence-bert", model_name: Optional[str] = None, device: str = "cuda"):
    """
    Factory function to create text encoder.
    
    Args:
        encoder_type: One of 'clip', 'sentence-bert', 't5', 'longformer'
        model_name: Specific model name (optional, uses defaults if None)
        device: Device to load model on
    
    Returns:
        Text encoder module
    
    Examples:
        # Sentence-BERT (recommended for long text)
        encoder = create_text_encoder("sentence-bert", "all-mpnet-base-v2")
        
        # T5 encoder
        encoder = create_text_encoder("t5", "t5-base")
        
        # Longformer for very long text
        encoder = create_text_encoder("longformer")
    """
    if encoder_type.lower() == "clip":
        model_name = model_name or "ViT-L-14"
        return CLIPTextEncoder(model_name=model_name, device=device)
    
    elif encoder_type.lower() == "sentence-bert" or encoder_type.lower() == "sbert":
        model_name = model_name or "all-mpnet-base-v2"  # Best quality
        # Alternative: "all-MiniLM-L6-v2" for faster, smaller (384 dim)
        return SentenceBERTEncoder(model_name=model_name, device=device)
    
    elif encoder_type.lower() == "t5":
        model_name = model_name or "t5-base"
        # Alternative: "t5-small" for smaller model (512 dim)
        return T5Encoder(model_name=model_name, device=device)
    
    elif encoder_type.lower() == "longformer":
        model_name = model_name or "allenai/longformer-base-4096"
        return LongformerEncoder(model_name=model_name, device=device)
    
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}. Choose from: clip, sentence-bert, t5, longformer")


# Recommended configurations for different use cases
RECOMMENDED_ENCODERS = {
    "long_text_high_quality": ("sentence-bert", "all-mpnet-base-v2", 768),  # Best for long text
    "long_text_fast": ("sentence-bert", "all-MiniLM-L6-v2", 384),  # Faster alternative
    "very_long_text": ("longformer", "allenai/longformer-base-4096", 768),  # For very long documents
    "understanding": ("t5", "t5-base", 768),  # Good for understanding semantics
    "original": ("clip", "ViT-L-14", 768),  # Original CLIP (77 tokens max)
}
