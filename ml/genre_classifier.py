"""
GENRE ACCURACY - Multi-Label Classifier (Random Forest)
Goal: Extract meaningful genre classifications from partial or noisy tags
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import hamming_loss
from typing import Dict, List, Tuple
import joblib
import logging

logger = logging.getLogger(__name__)


class GenreClassifier:
    """
    Multi-label genre classification using Random Forest.
    
    Inputs:
    - Audio features (energy, valence, danceability, tempo, acousticness)
    - Artist genre tags (from Spotify)
    
    Output:
    - Genre probability vector per song
    """
    
    def __init__(self, max_depth: int = 15, n_estimators: int = 100):
        """
        Initialize genre classifier.
        
        Args:
            max_depth: Max tree depth
            n_estimators: Number of trees in forest
        """
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        
        self.model = RandomForestClassifier(
            max_depth=max_depth,
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1
        )
        
        self.mlb = MultiLabelBinarizer()
        self.feature_cols = [
            'energy', 'valence', 'danceability', 
            'tempo', 'acousticness', 'loudness', 
            'speechiness', 'instrumentalness'
        ]
        self.genre_list = []
        self.fitted = False
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and normalize audio features."""
        
        feature_df = df[self.feature_cols].copy()
        
        # Fill missing values
        feature_df = feature_df.fillna(feature_df.mean())
        
        # Normalize tempo
        feature_df['tempo'] = feature_df['tempo'] / 200.0
        
        return feature_df
    
    def train(self, df: pd.DataFrame) -> None:
        """
        Train multi-label genre classifier.
        
        Args:
            df: DataFrame with 'genres' (list) and audio features
        """
        logger.info("Training genre classifier...")
        
        # Prepare features
        X = self.prepare_features(df)
        
        # Prepare multi-label targets
        genre_lists = df['genres'].tolist()
        y = self.mlb.fit_transform(genre_lists)
        
        self.genre_list = list(self.mlb.classes_)
        
        # Train model
        self.model.fit(X, y)
        
        self.fitted = True
        logger.info(f"Genre classifier trained with {len(self.genre_list)} genres")
    
    def predict_genre_probabilities(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get genre probability vector for each track.
        
        Args:
            df: DataFrame with audio features
            
        Returns:
            DataFrame with genre probabilities
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted yet")
        
        X = self.prepare_features(df)
        
        # Get probabilities
        probabilities = self.model.predict_proba(X)
        
        # Create results DataFrame
        results = pd.DataFrame(
            probabilities,
            columns=self.genre_list
        )
        
        return results
    
    def get_top_genres(self, df: pd.DataFrame, n: int = 5) -> List[str]:
        """
        Get top genres for a set of tracks.
        
        Args:
            df: DataFrame with audio features
            n: Number of top genres
            
        Returns:
            List of top genre names
        """
        genre_probs = self.predict_genre_probabilities(df)
        
        # Average probabilities across tracks
        avg_probs = genre_probs.mean().sort_values(ascending=False)
        
        return avg_probs.head(n).index.tolist()
    
    def get_weighted_genre_vector(self, df: pd.DataFrame, 
                                  weights: pd.Series = None) -> Dict[str, float]:
        """
        Get user's genre preference vector (weighted by play count).
        
        Args:
            df: User's track data
            weights: Optional weights (e.g., play counts)
            
        Returns:
            Dict mapping genre → weighted score
        """
        if weights is None:
            weights = pd.Series([1.0] * len(df))
        
        genre_probs = self.predict_genre_probabilities(df)
        
        # Weight by play count
        weighted_genres = {}
        total_weight = weights.sum()
        
        for genre in self.genre_list:
            weighted_score = (genre_probs[genre] * weights).sum() / total_weight
            weighted_genres[genre] = float(weighted_score)
        
        return dict(sorted(weighted_genres.items(), 
                          key=lambda x: x[1], 
                          reverse=True))
    
    def save_model(self, filepath: str) -> None:
        """Save fitted model."""
        joblib.dump({
            'model': self.model,
            'mlb': self.mlb,
            'genre_list': self.genre_list,
            'fitted': self.fitted
        }, filepath)
        logger.info(f"Genre classifier saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load fitted model."""
        data = joblib.load(filepath)
        self.model = data['model']
        self.mlb = data['mlb']
        self.genre_list = data['genre_list']
        self.fitted = data['fitted']
        logger.info(f"Genre classifier loaded from {filepath}")


# Usage Example
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'track_name': [f'Song_{i}' for i in range(100)],
        'energy': np.random.uniform(0, 1, 100),
        'valence': np.random.uniform(0, 1, 100),
        'danceability': np.random.uniform(0, 1, 100),
        'tempo': np.random.uniform(60, 200, 100),
        'acousticness': np.random.uniform(0, 1, 100),
        'loudness': np.random.uniform(-20, 0, 100),
        'speechiness': np.random.uniform(0, 1, 100),
        'instrumentalness': np.random.uniform(0, 1, 100),
        'genres': [
            ['pop', 'dance'],
            ['rock', 'alternative'],
            ['hiphop', 'rap'],
            ['electronic', 'edm'],
            ['classical', 'instrumental']
        ] * 20,
        'play_count': np.random.randint(1, 50, 100)
    }
    
    df = pd.DataFrame(sample_data)
    
    # Train classifier
    classifier = GenreClassifier()
    classifier.train(df)
    
    # Get weighted genre vector for a user
    weighted_genres = classifier.get_weighted_genre_vector(
        df, 
        weights=df['play_count']
    )
    print("Top 10 Genres (weighted by play count):")
    for genre, score in list(weighted_genres.items())[:10]:
        print(f"  {genre}: {score:.4f}")
