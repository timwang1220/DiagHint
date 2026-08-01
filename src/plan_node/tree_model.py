# tree_model.py
# Tree-LSTM model for plan similarity learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class BinaryTreeLSTM(nn.Module):
    """
    Binary Tree LSTM for encoding query plans.

    Based on the implementation from src/judger/model.py but adapted
    for sentence-transformer features (1163-dim instead of 4-dim).
    """

    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim

        # Define layers
        self.W_iou = nn.Linear(feature_dim, 3 * hidden_dim)
        self.U_iou = nn.Linear(2 * hidden_dim, 3 * hidden_dim)
        self.W_f = nn.Linear(feature_dim, 2 * hidden_dim)
        self.U_f = nn.Linear(2 * hidden_dim, 2 * hidden_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'W_' in name:
                    # Input to gate: small Xavier
                    nn.init.xavier_uniform_(param, gain=0.5)
                elif 'U_' in name:
                    # Hidden to gate: orthogonal
                    nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

                # Forget gate bias = 1
                if 'W_f.bias' in name or 'U_f.bias' in name:
                    param.data.fill_(1.0)

    def forward(self, features_list: List[torch.Tensor], children_index_list: List[torch.Tensor]):
        """
        Forward pass supporting batch processing.

        Args:
            features_list: List of (N_i, feature_dim) tensors
            children_index_list: List of (N_i, 2) tensors

        Returns:
            Batch of tree embeddings: (batch_size, hidden_dim)
        """
        batch_size = len(features_list)
        batch_output = []

        for features, children_index in zip(features_list, children_index_list):
            output = self._forward_single_tree(features, children_index)
            batch_output.append(output)

        return torch.stack(batch_output, dim=0)

    def _forward_single_tree(self, features: torch.Tensor, children_index: torch.Tensor):
        """
        Process a single tree.

        Args:
            features: (N, feature_dim) tensor
            children_index: (N, 2) tensor with child indices, -1 for missing

        Returns:
            Root node hidden state: (hidden_dim,)
        """
        N = features.size(0)
        device = features.device

        h = torch.zeros(N, self.hidden_dim, device=device)
        c = torch.zeros(N, self.hidden_dim, device=device)

        # Topological order: process children before parents
        order = self._topo_order(children_index)

        for i in order:
            x = features[i].unsqueeze(0)  # (1, feature_dim)

            left, right = children_index[i].tolist()
            hl = h[left].unsqueeze(0).clone() if left != -1 else torch.zeros(1, self.hidden_dim, device=device)
            hr = h[right].unsqueeze(0).clone() if right != -1 else torch.zeros(1, self.hidden_dim, device=device)
            cl = c[left].unsqueeze(0).clone() if left != -1 else torch.zeros(1, self.hidden_dim, device=device)
            cr = c[right].unsqueeze(0).clone() if right != -1 else torch.zeros(1, self.hidden_dim, device=device)

            hc = torch.cat([hl, hr], dim=-1)

            iou = self.W_iou(x) + self.U_iou(hc)
            input_gate, o, u = torch.chunk(iou, 3, dim=-1)
            input_gate = torch.sigmoid(input_gate)
            o = torch.sigmoid(o)
            u = torch.tanh(u)

            f = self.W_f(x) + self.U_f(hc)
            fl, fr = torch.chunk(torch.sigmoid(f), 2, dim=-1)

            c[i] = torch.clamp(
                input_gate * u + fl * cl + fr * cr,
                min=-10.0,
                max=10.0
            )

            h[i] = o * torch.tanh(c[i])

        root = order[-1]  # Last processed is the root
        return h[root]

    def _topo_order(self, children: torch.Tensor) -> List[int]:
        """
        Compute topological order for tree processing (post-order traversal).

        Args:
            children: (N, 2) tensor

        Returns:
            List of node indices in post-order
        """
        n = len(children)
        has_parent = [False] * n

        for i in range(n):
            l = int(children[i][0].item())
            r = int(children[i][1].item())
            if l != -1:
                has_parent[l] = True
            if r != -1:
                has_parent[r] = True

        roots = [i for i in range(n) if not has_parent[i]]

        visited = set()
        order = []

        def dfs(x: int):
            if x in visited or x == -1:
                return
            visited.add(x)
            l = int(children[x][0].item())
            r = int(children[x][1].item())
            dfs(l)
            dfs(r)
            order.append(x)

        for root in roots:
            dfs(root)

        return order


class TreeSimilarityModel(nn.Module):
    """
    Model for computing similarity between two query plans.

    Uses BinaryTreeLSTM to encode each plan, then computes similarity.
    """

    def __init__(
        self,
        feature_dim: int = 1163,  # 6 + 384*3 + 5
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # Tree encoder
        self.tree_encoder = BinaryTreeLSTM(feature_dim, hidden_dim)

        # Similarity head
        self.similarity_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, trees1: Tuple, trees2: Tuple) -> torch.Tensor:
        """
        Compute similarity between pairs of trees.

        Args:
            trees1: Tuple of (features_list, children_index_list) for first trees
            trees2: Tuple of (features_list, children_index_list) for second trees

        Returns:
            Similarity scores: (batch_size, 1)
        """
        features_list1, children_list1 = trees1
        features_list2, children_list2 = trees2

        # Encode trees
        embeddings1 = self.tree_encoder(features_list1, children_list1)
        embeddings2 = self.tree_encoder(features_list2, children_list2)

        # Concatenate embeddings
        combined = torch.cat([embeddings1, embeddings2], dim=1)

        # Compute similarity
        similarity = self.similarity_head(combined)

        return similarity

    def encode_single_tree(self, features: torch.Tensor, children_index: torch.Tensor) -> torch.Tensor:
        """
        Encode a single tree.

        Args:
            features: (N, feature_dim) tensor
            children_index: (N, 2) tensor

        Returns:
            Tree embedding: (hidden_dim,)
        """
        return self.tree_encoder([features], [children_index])[0]


class ContrastiveTreeModel(nn.Module):
    """
    Model for contrastive learning on tree pairs.

    Uses binary cross-entropy loss for similarity prediction.
    """

    def __init__(
        self,
        feature_dim: int = 1163,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        temperature: float = 0.5,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.temperature = temperature

        # Tree encoder
        self.tree_encoder = BinaryTreeLSTM(feature_dim, hidden_dim)

        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

    def forward(self, trees1: Tuple, trees2: Tuple) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for contrastive learning.

        Args:
            trees1: Tuple of (features_list, children_index_list)
            trees2: Tuple of (features_list, children_index_list)

        Returns:
            (similarity_scores, embeddings1, embeddings2)
        """
        features_list1, children_list1 = trees1
        features_list2, children_list2 = trees2

        # Encode trees
        raw_embeddings1 = self.tree_encoder(features_list1, children_list1)
        raw_embeddings2 = self.tree_encoder(features_list2, children_list2)

        # Project
        embeddings1 = self.projection_head(raw_embeddings1)
        embeddings2 = self.projection_head(raw_embeddings2)

        # Compute cosine similarity
        similarity = F.cosine_similarity(embeddings1, embeddings2, dim=1, keepdim=True)

        # Scale to [0, 1] range
        similarity = (similarity + 1) / 2

        return similarity, embeddings1, embeddings2


class TreePairClassifier(nn.Module):
    """
    Binary classifier for tree pairs.

    Returns probability that two trees are similar (same query).
    """

    def __init__(
        self,
        feature_dim: int = 1163,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # Tree encoder
        self.tree_encoder = BinaryTreeLSTM(feature_dim, hidden_dim)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # emb1, emb2, |emb1 - emb2|
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, trees1: Tuple, trees2: Tuple) -> torch.Tensor:
        """
        Classify tree pairs.

        Args:
            trees1: Tuple of (features_list, children_index_list)
            trees2: Tuple of (features_list, children_index_list)

        Returns:
            Probability of being similar: (batch_size, 1)
        """
        features_list1, children_list1 = trees1
        features_list2, children_list2 = trees2

        # Encode trees
        embeddings1 = self.tree_encoder(features_list1, children_list1)
        embeddings2 = self.tree_encoder(features_list2, children_list2)

        # Compute absolute difference
        diff = torch.abs(embeddings1 - embeddings2)

        # Concatenate
        combined = torch.cat([embeddings1, embeddings2, diff], dim=1)

        # Classify
        probability = self.classifier(combined)

        return probability


if __name__ == "__main__":
    # Test the models
    feature_dim = 1163
    hidden_dim = 256
    batch_size = 4

    # Create dummy data
    # Each tree has 3 nodes
    features_list = [
        torch.randn(3, feature_dim) for _ in range(batch_size)
    ]
    # Binary tree: 0 -> [1, 2], 1 -> [-1, -1], 2 -> [-1, -1]
    children_list = [
        torch.tensor([[1, 2], [-1, -1], [-1, -1]], dtype=torch.long)
        for _ in range(batch_size)
    ]

    print("Testing BinaryTreeLSTM...")
    tree_lstm = BinaryTreeLSTM(feature_dim, hidden_dim)
    output = tree_lstm(features_list, children_list)
    print(f"  Output shape: {output.shape}")  # Should be (4, 256)

    print("\nTesting TreeSimilarityModel...")
    sim_model = TreeSimilarityModel(feature_dim, hidden_dim)
    similarity = sim_model((features_list, children_list), (features_list, children_list))
    print(f"  Similarity shape: {similarity.shape}")  # Should be (4, 1)

    print("\nTesting ContrastiveTreeModel...")
    contrast_model = ContrastiveTreeModel(feature_dim, hidden_dim)
    sim, emb1, emb2 = contrast_model((features_list, children_list), (features_list, children_list))
    print(f"  Similarity shape: {sim.shape}")  # Should be (4, 1)
    print(f"  Embeddings1 shape: {emb1.shape}")  # Should be (4, 128)
    print(f"  Embeddings2 shape: {emb2.shape}")  # Should be (4, 128)

    print("\nTesting TreePairClassifier...")
    classifier = TreePairClassifier(feature_dim, hidden_dim)
    prob = classifier((features_list, children_list), (features_list, children_list))
    print(f"  Probability shape: {prob.shape}")  # Should be (4, 1)

    print("\nAll tests passed!")
