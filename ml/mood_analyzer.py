"""
Mood Analysis Module
Objective: Identify latent user mood preferences using K-Means clustering on audio features.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Dict, Tuple, List
import joblib
import logging

logger = logging.getLogger(__name__)


class MoodAnalyzer:
    """
    K-Means clustering on Spotify audio features to identify user mood preferences.
    
    Audio Features Used:
    - Energy: Intensity & activity (0.0 to 1.0)
    - Valence: Musical positiveness (0.0 to 1.0)
    - Danceability: How suitable for dancing (0.0 to 1.0)
    - Tempo: Beats per minute
    - Acousticness: Confidence measure of acoustic (0.0 to 1.0)
    """
    
    MOOD_NAMES = {
        'high_energy_happy': 'High Energy & Happy',
        'chill_mellow': 'Chill & Mellow',
        'dark_intense': 'Dark & Intense',
        'acoustic_folk': 'Acoustic & Folk',
        'upbeat_dance': 'Upbeat & Dance',
        'sad_melancholic': 'Sad & Melancholic'
    }
    
    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        """
        Initialize mood analyzer.
        
        Args:
            n_clusters: Number of mood clusters (3-6 recommended)
            random_state: Random seed
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
            max_iter=300
        )
        
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)  # For visualization
        
        self.feature_cols = [
            'energy', 'valence', 'danceability', 
            'tempo', 'acousticness'
        ]
        
        self.cluster_profiles = {}
        self.fitted = False
    
    def prepare_audio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare and validate audio features.
        
        Args:
            df: DataFrame with audio features
            
        Returns:
            Cleaned DataFrame with required features
        """
        logger.info("Preparing audio features...")
        
        # Ensure all required columns exist
        missing_cols = set(self.feature_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing audio features: {missing_cols}")
        
        # Select feature columns
        feature_df = df[self.feature_cols].copy()
        
        # Handle missing values
        feature_df = feature_df.fillna(feature_df.mean())
        
        # Normalize tempo to 0-1 range (typical range: 0-200 BPM)
        feature_df['tempo_normalized'] = feature_df['tempo'] / 200.0
        feature_df['tempo_normalized'] = feature_df['tempo_normalized'].clip(0, 1)
        
        # Update feature columns to use normalized tempo
        feature_cols_to_use = [
            'energy', 'valence', 'danceability', 
            'tempo_normalized', 'acousticness'
        ]
        
        return feature_df[feature_cols_to_use]
    
    def fit(self, df: pd.DataFrame) -> np.ndarray:
        """
        Train K-Means clustering on user's tracks.
        
        Args:
            df: DataFrame with audio features and optional play_count
            
        Returns:
            Cluster assignments for each track
        """
        logger.info("Fitting mood clusters...")
        
        # Prepare features
        feature_df = self.prepare_audio_features(df)
        
        # Normalize features
        features_scaled = self.scaler.fit_transform(feature_df)
        
        # Fit K-Means
        cluster_labels = self.kmeans.fit_predict(features_scaled)
        
        # Build cluster profiles
        df_with_clusters = df.copy()
        df_with_clusters['cluster'] = cluster_labels
        
        self._build_cluster_profiles(df_with_clusters)
        
        self.fitted = True
        logger.info(f"Fitted {self.n_clusters} mood clusters")
        
        return cluster_labels
    
    def _build_cluster_profiles(self, df: pd.DataFrame) -> None:
        """Build interpretable profiles for each cluster."""
        
        for cluster_id in range(self.n_clusters):
            cluster_data = df[df['cluster'] == cluster_id]
            
            profile = {
                'cluster_id': cluster_id,
                'size': len(cluster_data),
                'avg_energy': float(cluster_data['energy'].mean()),
                'avg_valence': float(cluster_data['valence'].mean()),
                'avg_danceability': float(cluster_data['danceability'].mean()),
                'avg_tempo': float(cluster_data['tempo'].mean()),
                'avg_acousticness': float(cluster_data['acousticness'].mean()),
                'label': self._assign_cluster_label(cluster_data)
            }
            
            self.cluster_profiles[cluster_id] = profile
    
    def _assign_cluster_label(self, cluster_data: pd.DataFrame) -> str:
        """Assign interpretable label to cluster based on characteristics."""
        
        avg_energy = cluster_data['energy'].mean()
        avg_valence = cluster_data['valence'].mean()
        avg_danceability = cluster_data['danceability'].mean()
        avg_acousticness = cluster_data['acousticness'].mean()
        
        # Decision logic
        if avg_energy > 0.7 and avg_valence > 0.6:
            return 'high_energy_happy'
        elif avg_energy < 0.4 and avg_valence < 0.4:
            return 'sad_melancholic'
        elif avg_acousticness > 0.6:
            return 'acoustic_folk'
        elif avg_danceability > 0.7:
            return 'upbeat_dance'
        elif avg_energy > 0.6 and avg_valence < 0.4:
            return 'dark_intense'
        else:
            return 'chill_mellow'
    
    def get_user_mood_distribution(self, df: pd.DataFrame, 
                                   weights: pd.Series = None) -> Dict[str, float]:
        """
        Get mood distribution for user based on listening data.
        
        Args:
            df: User's track data with audio features
            weights: Optional weights for tracks (e.g., play counts)
            
        Returns:
            Distribution dict: {mood_name: percentage}
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted yet")
        
        # Prepare and predict
        feature_df = self.prepare_audio_features(df)
        features_scaled = self.scaler.transform(feature_df)
        clusters = self.kmeans.predict(features_scaled)
        
        # Weight by play count if provided
        if weights is None:
            weights = pd.Series([1.0] * len(df))
        
        # Calculate distribution
        distribution = {}
        total_weight = weights.sum()
        
        for cluster_id in range(self.n_clusters):
            mask = clusters == cluster_id
            cluster_weight = weights[mask].sum()
            percentage = (cluster_weight / total_weight * 100) if total_weight > 0 else 0
            
            label = self.cluster_profiles[cluster_id]['label']
            distribution[label] = float(percentage)
        
        return distribution
    
    def get_dominant_mood(self, df: pd.DataFrame, 
                         weights: pd.Series = None) -> Dict:
        """
        Get dominant mood cluster for user.
        
        Args:
            df: User's track data
            weights: Optional weights
            
        Returns:
            Dict with mood info
        """
        distribution = self.get_user_mood_distribution(df, weights)
        
        dominant_mood = max(distribution.items(), key=lambda x: x[1])
        
        return {
            'dominant_mood': dominant_mood[0],
            'percentage': dominant_mood[1],
            'full_distribution': distribution
        }
    
    def get_mood_insights(self, df: pd.DataFrame, 
                         weights: pd.Series = None) -> Dict:
        """
        Generate human-readable mood insights.
        
        Args:
            df: User's track data
            weights: Optional weights
            
        Returns:
            Detailed insights dict
        """
        distribution = self.get_user_mood_distribution(df, weights)
        dominant = max(distribution.items(), key=lambda x: x[1])
        
        # Generate insight sentences
        insights = []
        if dominant[1] > 70:
            insights.append(f"You're primarily a {dominant[0].lower()} listener ({dominant[1]:.1f}% of your music)")
        else:
            insights.append(f"You have diverse listening habits, with {dominant[0].lower()} being most common ({dominant[1]:.1f}%)")
        
        sorted_moods = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_moods) > 1:
            insights.append(f"Your second preference is {sorted_moods[1][0].lower()} ({sorted_moods[1][1]:.1f}%)")
        
        # Feature-based insight
        avg_energy = df['energy'].mean()
        if avg_energy > 0.7:
            insights.append(f"You prefer high-energy music (avg energy: {avg_energy:.2f})")
        elif avg_energy < 0.4:
            insights.append(f"You prefer calm music (avg energy: {avg_energy:.2f})")
        
        avg_valence = df['valence'].mean()
        if avg_valence > 0.6:
            insights.append(f"Your music is generally uplifting and positive (avg valence: {avg_valence:.2f})")
        elif avg_valence < 0.4:
            insights.append(f"You gravitate towards emotional, introspective music (avg valence: {avg_valence:.2f})")
        
        return {
            'distribution': distribution,
            'dominant_mood': dominant[0],
            'insights': insights,
            'cluster_profiles': self.cluster_profiles
        }
    
    def get_visualization_data(self, df: pd.DataFrame) -> Dict:
        """Get 2D PCA data for visualization."""
        
        feature_df = self.prepare_audio_features(df)
        features_scaled = self.scaler.transform(feature_df)
        
        pca_data = self.pca.fit_transform(features_scaled)
        clusters = self.kmeans.predict(features_scaled)
        
        return {
            'x': pca_data[:, 0].tolist(),
            'y': pca_data[:, 1].tolist(),
            'clusters': clusters.tolist()
        }
    
    def save_model(self, filepath: str) -> None:
        """Save fitted model."""
        joblib.dump({
            'kmeans': self.kmeans,
            'scaler': self.scaler,
            'pca': self.pca,
            'cluster_profiles': self.cluster_profiles,
            'fitted': self.fitted
        }, filepath)
        logger.info(f"Mood model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load fitted model."""
        data = joblib.load(filepath)
        self.kmeans = data['kmeans']
        self.scaler = data['scaler']
        self.pca = data['pca']
        self.cluster_profiles = data['cluster_profiles']
        self.fitted = data['fitted']
        logger.info(f"Mood model loaded from {filepath}")


# Usage Example
if __name__ == "__main__":
    # Create sample track data with audio features
    np.random.seed(42)
    
    sample_tracks = {
        'track_name': [f'Song_{i}' for i in range(200)],
        'energy': np.random.uniform(0, 1, 200),
        'valence': np.random.uniform(0, 1, 200),
        'danceability': np.random.uniform(0, 1, 200),
        'tempo': np.random.uniform(60, 200, 200),
        'acousticness': np.random.uniform(0, 1, 200),
        'play_count': np.random.randint(1, 20, 200)
    }
    
    df = pd.DataFrame(sample_tracks)
    
    # Train mood analyzer
    analyzer = MoodAnalyzer(n_clusters=4)
    clusters = analyzer.fit(df)
    
    # Get mood distribution
    mood_dist = analyzer.get_user_mood_distribution(df, weights=df['play_count'])
    print("Mood Distribution:", mood_dist)
    
    # Get insights
    insights = analyzer.get_mood_insights(df, weights=df['play_count'])
    print("\nMood Insights:")
    for insight in insights['insights']:
        print(f"  - {insight}")
