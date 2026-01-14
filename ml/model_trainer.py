"""
COMPLETE MODEL TRAINING PIPELINE
Load data, train all models, save them for inference
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import sys

# Assuming models are in the same directory
# In production: from app.ml.taste_matcher import TasteMatcher, etc.

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SpotifyWrappedTrainer:
    """
    Complete training pipeline for Spotify Wrapped AI/ML models.
    """
    
    def __init__(self, data_path: str, models_path: str = "models"):
        """
        Initialize trainer.
        
        Args:
            data_path: Path to CSV with track data
            models_path: Directory to save trained models
        """
        self.data_path = data_path
        self.models_path = Path(models_path)
        self.models_path.mkdir(exist_ok=True)
        
        self.tracks_df = None
        self.interactions_df = None
        
        self.taste_matcher = None
        self.mood_analyzer = None
        self.genre_classifier = None
        self.recommender = None
    
    def load_data(self):
        """
        Load track data and interaction data.
        
        Expected track CSV columns:
        - track_uri, track_name, album_name, artist_names
        - energy, valence, danceability, tempo, acousticness
        - loudness, speechiness, instrumentalness, liveness
        - mode, key, time_signature, duration_ms
        - explicit, popularity, genres, release_date
        """
        logger.info(f"Loading track data from {self.data_path}")
        
        self.tracks_df = pd.read_csv(self.data_path)
        
        # Data validation
        required_cols = [
            'track_uri', 'track_name', 'energy', 'valence',
            'danceability', 'tempo', 'acousticness'
        ]
        
        missing_cols = set(required_cols) - set(self.tracks_df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        logger.info(f"Loaded {len(self.tracks_df)} tracks")
        
        # Parse genres if string
        if self.tracks_df['genres'].dtype == 'object':
            self.tracks_df['genres'] = self.tracks_df['genres'].apply(
                lambda x: eval(x) if isinstance(x, str) else x
            )
        
        return self.tracks_df
    
    def load_interactions(self, interactions_path: str):
        """
        Load user-track interaction data.
        
        Expected columns:
        - user_id, track_id, interaction_type, count, timestamp
        """
        logger.info(f"Loading interaction data from {interactions_path}")
        
        self.interactions_df = pd.read_csv(interactions_path)
        logger.info(f"Loaded {len(self.interactions_df)} interactions")
        
        return self.interactions_df
    
    def train_taste_matcher(self):
        """Train ALS matrix factorization model."""
        
        logger.info("=" * 60)
        logger.info("TRAINING TASTE MATCHER (ALS Matrix Factorization)")
        logger.info("=" * 60)
        
        if self.interactions_df is None:
            raise RuntimeError("Interactions data not loaded. Call load_interactions() first.")
        
        # Import here to avoid circular imports
        from taste_matcher import TasteMatcher
        
        self.taste_matcher = TasteMatcher(factors=50, iterations=15)
        
        # Prepare interaction matrix
        interaction_matrix, user_map, track_map = \
            self.taste_matcher.prepare_interaction_matrix(self.interactions_df)
        
        logger.info(f"Interaction matrix shape: {interaction_matrix.shape}")
        logger.info(f"Users: {len(user_map)}, Tracks: {len(track_map)}")
        
        # Train model
        self.taste_matcher.train(interaction_matrix)
        
        # Get embeddings
        user_embs, track_embs = self.taste_matcher.get_embeddings()
        logger.info(f"User embeddings shape: {user_embs.shape}")
        logger.info(f"Track embeddings shape: {track_embs.shape}")
        
        # Save model
        model_path = self.models_path / "taste_matcher.pkl"
        self.taste_matcher.save_model(str(model_path))
        logger.info(f"Taste matcher saved to {model_path}")
    
    def train_mood_analyzer(self):
        """Train K-means mood clustering model."""
        
        logger.info("=" * 60)
        logger.info("TRAINING MOOD ANALYZER (K-Means Clustering)")
        logger.info("=" * 60)
        
        from mood_analyzer import MoodAnalyzer
        
        self.mood_analyzer = MoodAnalyzer(n_clusters=4)
        
        # Fit on all tracks
        clusters = self.mood_analyzer.fit(self.tracks_df)
        
        logger.info(f"Fitted {self.mood_analyzer.n_clusters} mood clusters")
        
        # Print cluster profiles
        for cluster_id, profile in self.mood_analyzer.cluster_profiles.items():
            logger.info(f"\nCluster {cluster_id}: {profile['label']}")
            logger.info(f"  Size: {profile['size']}")
            logger.info(f"  Avg Energy: {profile['avg_energy']:.2f}")
            logger.info(f"  Avg Valence: {profile['avg_valence']:.2f}")
            logger.info(f"  Avg Danceability: {profile['avg_danceability']:.2f}")
        
        # Save model
        model_path = self.models_path / "mood_analyzer.pkl"
        self.mood_analyzer.save_model(str(model_path))
        logger.info(f"\nMood analyzer saved to {model_path}")
    
    def train_genre_classifier(self):
        """Train multi-label genre classification model."""
        
        logger.info("=" * 60)
        logger.info("TRAINING GENRE CLASSIFIER (Random Forest)")
        logger.info("=" * 60)
        
        from genre_classifier import GenreClassifier
        
        self.genre_classifier = GenreClassifier(n_estimators=100)
        
        # Train on all tracks
        self.genre_classifier.train(self.tracks_df)
        
        logger.info(f"Trained on {len(self.genre_classifier.genre_list)} unique genres")
        logger.info(f"Top genres: {self.genre_classifier.genre_list[:10]}")
        
        # Save model
        model_path = self.models_path / "genre_classifier.pkl"
        self.genre_classifier.save_model(str(model_path))
        logger.info(f"\nGenre classifier saved to {model_path}")
    
    def train_all(self, interactions_path: str = None):
        """
        Train all models in sequence.
        
        Args:
            interactions_path: Path to interactions CSV (required for taste matcher)
        """
        try:
            # Load data
            self.load_data()
            
            # Train models
            if interactions_path:
                self.load_interactions(interactions_path)
                self.train_taste_matcher()
            
            self.train_mood_analyzer()
            self.train_genre_classifier()
            
            logger.info("\n" + "=" * 60)
            logger.info("ALL MODELS TRAINED SUCCESSFULLY!")
            logger.info("=" * 60)
            logger.info(f"Models saved to: {self.models_path.absolute()}")
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
    
    def generate_training_report(self) -> Dict:
        """Generate report on trained models."""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'tracks_count': len(self.tracks_df),
            'models': {}
        }
        
        if self.taste_matcher:
            report['models']['taste_matcher'] = {
                'type': 'ALS Matrix Factorization',
                'factors': self.taste_matcher.factors,
                'users': len(self.taste_matcher.user_id_map),
                'tracks': len(self.taste_matcher.track_id_map),
                'user_embedding_size': tuple(self.taste_matcher.user_embeddings.shape)
            }
        
        if self.mood_analyzer:
            report['models']['mood_analyzer'] = {
                'type': 'K-Means Clustering',
                'clusters': self.mood_analyzer.n_clusters,
                'fitted': self.mood_analyzer.fitted
            }
        
        if self.genre_classifier:
            report['models']['genre_classifier'] = {
                'type': 'Random Forest Multi-Label',
                'genres': len(self.genre_classifier.genre_list),
                'estimators': self.genre_classifier.n_estimators
            }
        
        return report


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage:
    
    python model_trainer.py --data ./data/SytheticData1000.csv --interactions ./data/interactions.csv
    """
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Spotify Wrapped AI/ML models")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to track data CSV"
    )
    parser.add_argument(
        "--interactions",
        type=str,
        help="Path to interactions CSV"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models",
        help="Output directory for trained models"
    )
    
    args = parser.parse_args()
    
    # Train models
    trainer = SpotifyWrappedTrainer(args.data, args.output)
    trainer.train_all(args.interactions)
    
    # Generate report
    report = trainer.generate_training_report()
    
    import json
    print("\n" + "=" * 60)
    print("TRAINING REPORT")
    print("=" * 60)
    print(json.dumps(report, indent=2))
