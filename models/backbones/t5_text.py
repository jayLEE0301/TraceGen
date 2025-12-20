import logging
from typing import List, Union

import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5EncoderModel

logger = logging.getLogger(__name__)


class T5TextEncoder(nn.Module):
    """
    T5 Text Encoder that extracts text features for conditioning point detection.

    Args:
        model_name: HuggingFace model identifier (e.g., 't5-small', 't5-base', 't5-large')
        freeze: Whether to freeze the text encoder parameters
        d_model: Output dimension (will add projection if different from T5 dim)
        max_length: Maximum text sequence length
        pooling_strategy: How to pool text features ('mean', 'attention', 'last')
    """

    def __init__(
        self,
        model_name: str = "t5-small",
        freeze: bool = True,
        d_model: int = 768,
        max_length: int = 128,
    ):
        super().__init__()

        self.model_name = model_name
        self.freeze = freeze
        self.d_model = d_model
        self.max_length = max_length

        # Load T5 encoder model and tokenizer
        logger.info(f"Loading T5 encoder model: {model_name}")
        self.text_model = T5EncoderModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Get the actual hidden size from the model
        self.t5_dim = self.text_model.config.d_model

        self.final_layernorm = nn.LayerNorm(self.t5_dim)

        # Add projection layer if needed
        if self.t5_dim != d_model:
            self.projection = nn.Linear(self.t5_dim, d_model)
        else:
            self.projection = nn.Identity()

        # Freeze parameters if requested
        if freeze:
            self._freeze_backbone()

        logger.info("T5 Text Encoder initialized:")
        logger.info(f"  - Model: {model_name}")
        logger.info(f"  - T5 dim: {self.t5_dim}")
        logger.info("  - Output dim: {d_model}")
        logger.info("  - Max length: {max_length}")
        logger.info("  - Frozen: {freeze}")

    def _freeze_backbone(self):
        """Freeze all text encoder parameters."""
        for param in self.text_model.parameters():
            param.requires_grad = False
        logger.info("T5 text backbone frozen")

    def unfreeze_last_blocks(self, num_blocks: int = 2):
        """
        Unfreeze the last N transformer blocks for fine-tuning.

        Args:
            num_blocks: Number of last blocks to unfreeze
        """
        if not self.freeze:
            logger.warning("Text backbone is not frozen, nothing to unfreeze")
            return

        # Unfreeze the last num_blocks encoder layers
        total_layers = len(self.text_model.encoder.block)
        start_idx = max(0, total_layers - num_blocks)

        for i in range(start_idx, total_layers):
            for param in self.text_model.encoder.block[i].parameters():
                param.requires_grad = True

        # Also unfreeze the final layer norm
        if hasattr(self.text_model.encoder, 'final_layer_norm'):
            for param in self.text_model.encoder.final_layer_norm.parameters():
                param.requires_grad = True

        logger.info(f"Unfroze last {num_blocks} T5 transformer blocks (layers {start_idx}-{total_layers-1})")

    def tokenize_texts(self, texts: Union[str, List[str]]) -> dict:
        """
        Tokenize input texts using T5 tokenizer.

        Args:
            texts: Single text string or list of text strings

        Returns:
            tokenized: Dictionary with input_ids and attention_mask
        """
        if isinstance(texts, str):
            texts = [texts]

        # Use the tokenizer to tokenize
        tokenized = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask']
        }

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the text encoder.

        Args:
            input_ids: Tokenized text input [B, 128]
            attention_mask: Attention mask [B, 128]

        Returns:
            text_features: Text sequence features [B, 128, d_model]
        """
        # Forward through T5 encoder
        outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_states = outputs.last_hidden_state  # [B, 128, hidden_size]

        # Post layer norm
        hidden_states = self.final_layernorm(hidden_states)
        # Project to target dimension
        text_features = self.projection(hidden_states)  # [B, 128, d_model]
        return text_features

    def encode_texts(self, texts: Union[str, List[str]], device: torch.device) -> torch.Tensor:
        """
        Convenient method to encode texts from strings.

        Args:
            texts: Single text string or list of text strings
            device: Target device

        Returns:
            text_features: Encoded text sequence features [B, 128, d_model]
        """
        # Tokenize
        tokenized = self.tokenize_texts(texts)
        input_ids = tokenized['input_ids'].to(device)
        attention_mask = tokenized['attention_mask'].to(device)

        # Encode
        with torch.no_grad() if self.freeze else torch.enable_grad():
            text_features = self.forward(input_ids, attention_mask)

        return text_features  # [B, 128, d_model]

    def get_text_config(self) -> dict:
        """Get text encoder configuration."""
        return {
            'vocab_size': self.text_model.config.vocab_size,
            'hidden_size': self.t5_dim,
            'num_layers': self.text_model.config.num_layers,
            'num_heads': self.text_model.config.num_heads,
            'max_length': self.max_length
        }
