# embedding.py
# TextEncoder class using sentence-transformers for encoding table names, columns, and predicates
import os
from typing import List, Dict, Optional
import numpy as np
from collections import defaultdict


class TextEncoder:
    """
    Wrapper for sentence-transformers model to encode text features.
    Uses caching to avoid recomputing embeddings for the same text.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: Optional[str] = None):
        """
        Initialize the text encoder.

        Args:
            model_name: Name of the sentence-transformers model
            device: Device to use ('cpu', 'cuda', or None for auto)
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.embedding_dim = 384  # all-MiniLM-L6-v2 outputs 384 dimensions

        # Caches for embeddings
        self._table_cache: Dict[str, np.ndarray] = {}
        self._column_cache: Dict[str, np.ndarray] = {}
        self._predicate_cache: Dict[str, np.ndarray] = {}

    def _load_model(self):
        """Lazy load the model."""
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, device=self.device)
            # Set model to eval mode
            if hasattr(self.model, 'eval'):
                self.model.eval()

    @property
    def dim(self) -> int:
        """Return the embedding dimension."""
        return self.embedding_dim

    def encode_table_name(self, table_name: str, use_cache: bool = True) -> np.ndarray:
        """
        Encode a single table name.

        Args:
            table_name: Name of the table
            use_cache: Whether to use cached embeddings

        Returns:
            Embedding vector of shape (embedding_dim,)
        """
        if not table_name:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        # Check cache
        if use_cache and table_name in self._table_cache:
            return self._table_cache[table_name]

        self._load_model()

        # For table names, add context to improve embedding
        text = f"table: {table_name}"
        embedding = self.model.encode(text, convert_to_numpy=True)

        # Convert to float32
        embedding = embedding.astype(np.float32)

        # Cache
        if use_cache:
            self._table_cache[table_name] = embedding

        return embedding

    def encode_column_names(self, column_names: List[str], use_cache: bool = True) -> np.ndarray:
        """
        Encode multiple column names and return their average.

        Args:
            column_names: List of column names (e.g., ['t.id', 'mc.movie_id'])
            use_cache: Whether to use cached embeddings

        Returns:
            Average embedding vector of shape (embedding_dim,)
        """
        if not column_names:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        self._load_model()

        # Get embeddings for each column
        embeddings = []
        for col in column_names:
            if not col:
                continue

            # Check cache
            if use_cache and col in self._column_cache:
                embeddings.append(self._column_cache[col])
                continue

            # Add context for column names
            text = f"column: {col}"
            embedding = self.model.encode(text, convert_to_numpy=True)
            embedding = embedding.astype(np.float32)

            # Cache
            if use_cache:
                self._column_cache[col] = embedding

            embeddings.append(embedding)

        if not embeddings:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        # Return average
        avg_embedding = np.mean(embeddings, axis=0).astype(np.float32)
        return avg_embedding

    def encode_predicates(self, predicates: List[str], use_cache: bool = True) -> np.ndarray:
        """
        Encode predicates by concatenating them and encoding the result.

        Args:
            predicates: List of predicate strings (e.g., ['(t.id = mc.movie_id)', '(kind = "verified")'])
            use_cache: Whether to use cached embeddings

        Returns:
            Embedding vector of shape (embedding_dim,)
        """
        if not predicates:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        self._load_model()

        # Sort predicates for consistent caching (order shouldn't matter for meaning)
        sorted_preds = sorted(predicates)

        # Create cache key from sorted predicates
        cache_key = " || ".join(sorted_preds)
        if use_cache and cache_key in self._predicate_cache:
            return self._predicate_cache[cache_key]

        # Concatenate predicates with context
        text = "predicates: " + " ".join(predicates)

        # Encode
        embedding = self.model.encode(text, convert_to_numpy=True)
        embedding = embedding.astype(np.float32)

        # Cache
        if use_cache:
            self._predicate_cache[cache_key] = embedding

        return embedding

    def encode_text(self, text: str, prefix: str = "") -> np.ndarray:
        """
        Generic method to encode any text with an optional prefix.

        Args:
            text: Text to encode
            prefix: Optional prefix to add context

        Returns:
            Embedding vector of shape (embedding_dim,)
        """
        if not text:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        self._load_model()

        full_text = f"{prefix} {text}" if prefix else text
        embedding = self.model.encode(full_text, convert_to_numpy=True)
        return embedding.astype(np.float32)

    def clear_cache(self):
        """Clear all caches."""
        self._table_cache.clear()
        self._column_cache.clear()
        self._predicate_cache.clear()

    def get_cache_stats(self) -> Dict[str, int]:
        """Return cache statistics."""
        return {
            "table_cache": len(self._table_cache),
            "column_cache": len(self._column_cache),
            "predicate_cache": len(self._predicate_cache),
        }

    def save_cache(self, cache_dir: str):
        """Save cache to disk."""
        os.makedirs(cache_dir, exist_ok=True)

        np.save(os.path.join(cache_dir, "table_cache.npy"), self._table_cache)
        np.save(os.path.join(cache_dir, "column_cache.npy"), self._column_cache)
        np.save(os.path.join(cache_dir, "predicate_cache.npy"), self._predicate_cache)

        # Save keys for lookup
        import json
        with open(os.path.join(cache_dir, "table_keys.json"), "w") as f:
            json.dump(list(self._table_cache.keys()), f)
        with open(os.path.join(cache_dir, "column_keys.json"), "w") as f:
            json.dump(list(self._column_cache.keys()), f)
        with open(os.path.join(cache_dir, "predicate_keys.json"), "w") as f:
            json.dump(list(self._predicate_cache.keys()), f)

    def load_cache(self, cache_dir: str):
        """Load cache from disk."""
        import json

        # Load keys
        table_keys_path = os.path.join(cache_dir, "table_keys.json")
        column_keys_path = os.path.join(cache_dir, "column_keys.json")
        predicate_keys_path = os.path.join(cache_dir, "predicate_keys.json")

        if os.path.exists(table_keys_path):
            with open(table_keys_path, "r") as f:
                table_keys = json.load(f)
            table_cache = np.load(os.path.join(cache_dir, "table_cache.npy"), allow_pickle=True).item()
            self._table_cache = {k: table_cache[k] for k in table_keys if k in table_cache}

        if os.path.exists(column_keys_path):
            with open(column_keys_path, "r") as f:
                column_keys = json.load(f)
            column_cache = np.load(os.path.join(cache_dir, "column_cache.npy"), allow_pickle=True).item()
            self._column_cache = {k: column_cache[k] for k in column_keys if k in column_cache}

        if os.path.exists(predicate_keys_path):
            with open(predicate_keys_path, "r") as f:
                predicate_keys = json.load(f)
            predicate_cache = np.load(os.path.join(cache_dir, "predicate_cache.npy"), allow_pickle=True).item()
            self._predicate_cache = {k: predicate_cache[k] for k in predicate_keys if k in predicate_cache}


# Singleton instance for use across modules
_global_encoder: Optional[TextEncoder] = None


def get_encoder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: Optional[str] = None) -> TextEncoder:
    """Get or create the global TextEncoder instance."""
    global _global_encoder
    if _global_encoder is None:
        _global_encoder = TextEncoder(model_name=model_name, device=device)
    return _global_encoder


def reset_encoder():
    """Reset the global encoder instance."""
    global _global_encoder
    _global_encoder = None


if __name__ == "__main__":
    # Test the encoder
    encoder = TextEncoder()

    # Test table name encoding
    table_emb = encoder.encode_table_name("movie_info")
    print(f"Table embedding shape: {table_emb.shape}")

    # Test column name encoding
    col_emb = encoder.encode_column_names(["t.id", "mc.movie_id", "kind"])
    print(f"Column embedding shape: {col_emb.shape}")

    # Test predicate encoding
    pred_emb = encoder.encode_predicates(["(t.id = mc.movie_id)", "(kind = 'verified')"])
    print(f"Predicate embedding shape: {pred_emb.shape}")

    # Test cache
    print(f"Cache stats: {encoder.get_cache_stats()}")

    # Test semantic similarity
    col1 = encoder.encode_column_names(["movie_id"])
    col2 = encoder.encode_column_names(["title_id"])
    col3 = encoder.encode_column_names(["production_year"])

    # Compute cosine similarity
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    print(f"\nSimilarity (movie_id vs title_id): {cosine_sim(col1, col2):.4f}")
    print(f"Similarity (movie_id vs production_year): {cosine_sim(col1, col3):.4f}")
    print(f"Similarity (title_id vs production_year): {cosine_sim(col2, col3):.4f}")
