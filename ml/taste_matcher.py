"""
TASTE MATCHING - ALS Matrix Factorization with Weighted Implicit Feedback
Core: Latent factor analysis for user taste profiling based on behavioral signals
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from sklearn.preprocessing import MinMaxScaler
import joblib
from typing import Tuple, Dict, List
import logging

logger = logging.getLogger(__name__)


class TasteMatcher:
    """
    Matrix Factorization using ALS for implicit feedback signals.
    
    Weighting Logic:
    - Complete Stream: 1.0
    - Interaction (Save): +2.0
    - Interaction (Replay): +1.5
    - Partial Stream (<30s): -1.0
    """
    
    def __init__(self, factors: int = 50, iterations: int = 15, regularization: float = 0.01):
        """
        Initialize ALS model.
        
        Args:
            factors: Latent factor dimensions (user & track embeddings)
            iterations: Training iterations
            regularization: L2 regularization lambda
        """
        self.factors = factors
        self.iterations = iterations
        self.regularization = regularization
        
        self.model = AlternatingLeastSquares(
            factors=factors,
            iterations=iterations,
            regularization=regularization,
            random_state=42,
            calculate_training_loss=True,
            use_gpu=False  # Set True if GPU available
        )
        
        self.user_id_map = {}
        self.track_id_map = {}
        self.reverse_user_map = {}
        self.reverse_track_map = {}
        
        self.scaler = MinMaxScaler()
        self.user_embeddings = None
        self.track_embeddings = None
        
    def _get_interaction_weight(self, interaction_type: str) -> float:
        """
        Get weight for interaction type.
        
        Args:
            interaction_type: 'listen', 'save', 'replay', 'skip'
            
        Returns:
            Weight score
        """
        weights = {
            'listen': 1.0,
            'save': 2.0,
            'replay': 1.5,
            'skip': -1.0
        }
        return weights.get(interaction_type, 1.0)
    
    def prepare_interaction_matrix(self, 
                                   df: pd.DataFrame) -> Tuple[csr_matrix, Dict, Dict]:
        """
        Build user × track weighted interaction matrix.
        
        Args:
            df: DataFrame with columns [user_id, track_id, interaction_type, count]
            
        Returns:
            Tuple of (sparse_matrix, user_id_map, track_id_map)
        """
        logger.info("Preparing interaction matrix...")
        
        # Build ID mappings
        unique_users = df['user_id'].unique()
        unique_tracks = df['track_id'].unique()
        
        self.user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self.track_id_map = {tid: idx for idx, tid in enumerate(unique_tracks)}
        
        self.reverse_user_map = {idx: uid for uid, idx in self.user_id_map.items()}
        self.reverse_track_map = {idx: tid for tid, idx in self.track_id_map.items()}
        
        # Map IDs
        df = df.copy()
        df['user_idx'] = df['user_id'].map(self.user_id_map)
        df['track_idx'] = df['track_id'].map(self.track_id_map)
        
        # Calculate weighted scores
        df['weight'] = df['interaction_type'].apply(self._get_interaction_weight)
        df['score'] = df['weight'] * df['count']  # Multiply by interaction count
        
        # Aggregate by user-track pair (sum multiple interactions)
        interaction_agg = df.groupby(['user_idx', 'track_idx'])['score'].sum().reset_index()
        
        # Build sparse matrix
        n_users = len(self.user_id_map)
        n_tracks = len(self.track_id_map)
        
        interaction_matrix = csr_matrix(
            (interaction_agg['score'], 
             (interaction_agg['user_idx'], interaction_agg['track_idx'])),
            shape=(n_users, n_tracks)
        )
        
        logger.info(f"Matrix shape: {interaction_matrix.shape}, Sparsity: {1 - (interaction_agg.shape[0] / (n_users * n_tracks)):.4f}")
        
        return interaction_matrix, self.user_id_map, self.track_id_map
    
    def train(self, interaction_matrix: csr_matrix) -> None:
        """
        Train ALS model.
        
        Args:
            interaction_matrix: Sparse user × track interaction matrix
        """
        logger.info("Training ALS model...")
        self.model.fit(interaction_matrix)
        logger.info("Training complete")
    
    def get_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get trained user and track embeddings.
        
        Returns:
            Tuple of (user_embeddings, track_embeddings)
        """
        self.user_embeddings = self.model.user_factors
        self.track_embeddings = self.model.item_factors
        
        return self.user_embeddings, self.track_embeddings
    
    def get_user_embedding(self, user_id: str) -> np.ndarray:
        """Get embedding vector for specific user."""
        if user_id not in self.user_id_map:
            raise ValueError(f"User {user_id} not in training set")
        
        user_idx = self.user_id_map[user_id]
        return self.user_embeddings[user_idx]
    
    def get_track_embedding(self, track_id: str) -> np.ndarray:
        """Get embedding vector for specific track."""
        if track_id not in self.track_id_map:
            raise ValueError(f"Track {track_id} not in training set")
        
        track_idx = self.track_id_map[track_id]
        return self.track_embeddings[track_idx]
    
    def get_taste_match_score(self, user_id: str, track_id: str) -> float:
        """
        Calculate taste match score between user and track.
        
        Args:
            user_id: User identifier
            track_id: Track identifier
            
        Returns:
            Taste match score (dot product of embeddings)
        """
        try:
            user_emb = self.get_user_embedding(user_id)
            track_emb = self.get_track_embedding(track_id)
            
            score = np.dot(user_emb, track_emb)
            # Normalize to 0-1 range
            score = (score + 1) / 2
            
            return float(score)
        except ValueError:
            return 0.5  # Default neutral score
    
    def get_user_taste_profile(self, user_id: str) -> Dict:
        """
        Get comprehensive taste profile for user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with taste metrics
        """
        user_emb = self.get_user_embedding(user_id)
        
        return {
            'user_id': user_id,
            'embedding_dimension': len(user_emb),
            'embedding_magnitude': float(np.linalg.norm(user_emb)),
            'embedding_mean': float(np.mean(user_emb)),
            'embedding_std': float(np.std(user_emb)),
            'embedding_vector': user_emb.tolist()  # For storage
        }
    
    def get_similar_users(self, user_id: str, n: int = 10) -> List[Tuple[str, float]]:
        """
        Find similar users based on embedding similarity.
        
        Args:
            user_id: Reference user
            n: Number of similar users to return
            
        Returns:
            List of (user_id, similarity_score) tuples
        """
        user_emb = self.get_user_embedding(user_id)
        
        # Cosine similarity with all users
        similarities = np.dot(self.user_embeddings, user_emb)
        similarities = similarities / (np.linalg.norm(self.user_embeddings, axis=1) * np.linalg.norm(user_emb))
        
        # Get top-n (excluding self)
        top_indices = np.argsort(similarities)[-n-1:-1][::-1]
        
        results = []
        for idx in top_indices:
            sim_user_id = self.reverse_user_map[idx]
            if sim_user_id != user_id:
                results.append((sim_user_id, float(similarities[idx])))
        
        return results
    
    def save_model(self, filepath: str) -> None:
        """Save trained model to disk."""
        joblib.dump({
            'model': self.model,
            'user_id_map': self.user_id_map,
            'track_id_map': self.track_id_map,
            'reverse_user_map': self.reverse_user_map,
            'reverse_track_map': self.reverse_track_map,
            'user_embeddings': self.user_embeddings,
            'track_embeddings': self.track_embeddings
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load trained model from disk."""
        data = joblib.load(filepath)
        self.model = data['model']
        self.user_id_map = data['user_id_map']
        self.track_id_map = data['track_id_map']
        self.reverse_user_map = data['reverse_user_map']
        self.reverse_track_map = data['reverse_track_map']
        self.user_embeddings = data['user_embeddings']
        self.track_embeddings = data['track_embeddings']
        logger.info(f"Model loaded from {filepath}")


# Usage Example
if __name__ == "__main__":
    # Create sample interaction data
    np.random.seed(42)
    n_users = 100
    n_tracks = 500
    
    # Generate synthetic interaction data
    interactions = []
    for _ in range(2000):
        user_id = f"user_{np.random.randint(0, n_users)}"
        track_id = f"track_{np.random.randint(0, n_tracks)}"
        interaction_type = np.random.choice(['listen', 'save', 'replay', 'skip'], p=[0.6, 0.15, 0.15, 0.1])
        count = np.random.randint(1, 5)
        
        interactions.append({
            'user_id': user_id,
            'track_id': track_id,
            'interaction_type': interaction_type,
            'count': count
        })
    
    df = pd.DataFrame(interactions)
    
    # Train model
    taste_matcher = TasteMatcher(factors=20, iterations=10)
    matrix, user_map, track_map = taste_matcher.prepare_interaction_matrix(df)
    taste_matcher.train(matrix)
    
    # Get embeddings
    user_embs, track_embs = taste_matcher.get_embeddings()
    print(f"User embeddings shape: {user_embs.shape}")
    print(f"Track embeddings shape: {track_embs.shape}")
    
    # Get taste match score
    test_user = "user_0"
    test_track = "track_0"
    score = taste_matcher.get_taste_match_score(test_user, test_track)
    print(f"Taste match score ({test_user} → {test_track}): {score:.4f}")
    
    # Get similar users
    similar_users = taste_matcher.get_similar_users(test_user, n=5)
    print(f"Users similar to {test_user}: {similar_users}")
