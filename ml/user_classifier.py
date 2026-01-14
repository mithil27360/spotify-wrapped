"""
USER CLASSIFIER - Listening Personality Classification
Classify users into personality types based on listening behavior
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple
import joblib
import logging

logger = logging.getLogger(__name__)


class UserClassifier:
    """
    Classify users into listening personality types.
    
    User Types (assigned after clustering):
    - Explorer: High genre diversity, discovers new music
    - Loyalist: Low diversity, replays favorites
    - Night Owl: Primarily night listening
    - Party Animal: High energy, high danceability
    - Chill Seeker: Low energy, high acousticness
    - Mood Rider: High valence variance
    """
    
    USER_TYPE_LABELS = {
        'explorer': {
            'name': 'The Explorer',
            'emoji': '🌍',
            'description': 'You love discovering new sounds and exploring diverse genres.',
            'traits': ['Diverse taste', 'New music seeker', 'Genre hopper']
        },
        'loyalist': {
            'name': 'The Loyalist',
            'emoji': '💎',
            'description': 'You know what you love and stick to your favorites.',
            'traits': ['Repeat listener', 'Artist devoted', 'Comfort zone lover']
        },
        'night_owl': {
            'name': 'The Night Owl',
            'emoji': '🦉',
            'description': 'Your best playlists come alive after midnight.',
            'traits': ['Late night vibes', 'Quiet hours listener', 'Nocturnal curator']
        },
        'party_animal': {
            'name': 'The Party Animal',
            'emoji': '🎉',
            'description': 'High energy tracks fuel your playlists.',
            'traits': ['Energy seeker', 'Dance lover', 'Upbeat vibes']
        },
        'chill_seeker': {
            'name': 'The Chill Seeker',
            'emoji': '😌',
            'description': 'You prefer acoustic, mellow, and laid-back sounds.',
            'traits': ['Acoustic lover', 'Relaxed vibes', 'Low-key energy']
        },
        'mood_rider': {
            'name': 'The Mood Rider',
            'emoji': '🎢',
            'description': 'Your music matches your emotions - highs and lows.',
            'traits': ['Emotional listener', 'Varied moods', 'Expressive taste']
        }
    }
    
    def __init__(self, n_clusters: int = 6, random_state: int = 42):
        """
        Initialize user classifier.
        
        Args:
            n_clusters: Number of user type clusters
            random_state: Random seed
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        
        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10
        )
        self.scaler = StandardScaler()
        
        self.feature_cols = [
            'genre_diversity',
            'repeat_artist_ratio',
            'night_listener_ratio',
            'avg_energy',
            'avg_valence',
            'avg_danceability',
            'avg_acousticness',
            'explicit_ratio'
        ]
        
        self.cluster_labels = {}
        self.fitted = False
        
    def prepare_user_features(self, user_features: Dict) -> np.ndarray:
        """
        Prepare user features for classification.
        
        Args:
            user_features: Dictionary of user features
            
        Returns:
            Feature array
        """
        features = []
        for col in self.feature_cols:
            value = user_features.get(col, 0)
            features.append(float(value) if value is not None else 0)
        
        return np.array(features).reshape(1, -1)
    
    def classify_user(self, user_features: Dict) -> Dict:
        """
        Classify a user into a personality type.
        
        Uses rule-based classification based on dominant characteristics.
        
        Args:
            user_features: Dictionary of user features
            
        Returns:
            Classification result with type, description, and traits
        """
        # Extract key features
        genre_diversity = user_features.get('genre_diversity', 0)
        repeat_ratio = user_features.get('repeat_artist_ratio', 0)
        night_ratio = user_features.get('night_listener_ratio', 0)
        avg_energy = user_features.get('avg_energy', 0.5)
        avg_valence = user_features.get('avg_valence', 0.5)
        avg_danceability = user_features.get('avg_danceability', 0.5)
        avg_acousticness = user_features.get('avg_acousticness', 0.5)
        
        # Calculate valence variance if available
        valence_variance = user_features.get('valence_variance', 0)
        
        # Rule-based classification
        scores = {
            'explorer': genre_diversity * 3 + (1 - repeat_ratio) * 2,
            'loyalist': repeat_ratio * 3 + (1 - genre_diversity) * 2,
            'night_owl': night_ratio * 5,
            'party_animal': avg_energy * 2 + avg_danceability * 2 + avg_valence,
            'chill_seeker': avg_acousticness * 2 + (1 - avg_energy) * 2 + (1 - avg_danceability),
            'mood_rider': valence_variance * 5 if valence_variance > 0 else abs(avg_valence - 0.5) * 2
        }
        
        # Get dominant type
        user_type = max(scores, key=scores.get)
        type_info = self.USER_TYPE_LABELS[user_type]
        
        return {
            'type_id': user_type,
            'type_name': type_info['name'],
            'emoji': type_info['emoji'],
            'description': type_info['description'],
            'traits': type_info['traits'],
            'confidence': scores[user_type] / sum(scores.values()),
            'all_scores': scores
        }
    
    def calculate_percentiles(self, user_features: Dict, 
                              global_stats: Dict = None) -> Dict[str, float]:
        """
        Calculate user's percentile rankings.
        
        Args:
            user_features: User's feature dictionary
            global_stats: Optional global statistics for comparison
            
        Returns:
            Dictionary of percentile rankings
        """
        # Default global averages (can be calibrated with real data)
        if global_stats is None:
            global_stats = {
                'avg_energy': {'mean': 0.55, 'std': 0.15},
                'avg_valence': {'mean': 0.45, 'std': 0.15},
                'avg_danceability': {'mean': 0.58, 'std': 0.12},
                'genre_diversity': {'mean': 0.02, 'std': 0.01},
                'night_listener_ratio': {'mean': 0.15, 'std': 0.1},
                'repeat_artist_ratio': {'mean': 0.3, 'std': 0.15},
                'explicit_ratio': {'mean': 0.25, 'std': 0.15}
            }
        
        percentiles = {}
        
        for feature, stats in global_stats.items():
            user_value = user_features.get(feature, stats['mean'])
            # Calculate z-score and convert to percentile
            z_score = (user_value - stats['mean']) / stats['std'] if stats['std'] > 0 else 0
            # Convert to percentile (0-100)
            from scipy.stats import norm
            try:
                percentile = norm.cdf(z_score) * 100
            except:
                percentile = 50  # Default to median if scipy not available
            
            percentiles[feature] = round(percentile, 1)
        
        return percentiles
    
    def get_listening_insights(self, user_features: Dict, 
                               percentiles: Dict) -> List[str]:
        """
        Generate human-readable listening insights.
        
        Args:
            user_features: User feature dictionary
            percentiles: Percentile rankings
            
        Returns:
            List of insight strings
        """
        insights = []
        
        # Energy insight
        energy_pct = percentiles.get('avg_energy', 50)
        if energy_pct > 75:
            insights.append(f"🔥 You're in the top {100-energy_pct:.0f}% for high-energy listening!")
        elif energy_pct < 25:
            insights.append(f"😌 You prefer chill vibes - more mellow than {energy_pct:.0f}% of listeners")
        
        # Genre diversity
        diversity_pct = percentiles.get('genre_diversity', 50)
        if diversity_pct > 80:
            insights.append(f"🌍 Musical explorer! You're more diverse than {diversity_pct:.0f}% of listeners")
        elif diversity_pct < 20:
            insights.append(f"💎 You know what you love - focused taste")
        
        # Night listening
        night_pct = percentiles.get('night_listener_ratio', 50)
        if night_pct > 70:
            insights.append(f"🦉 Night owl alert! Top {100-night_pct:.0f}% for late-night listening")
        
        # Repeat listener
        repeat_pct = percentiles.get('repeat_artist_ratio', 50)
        if repeat_pct > 70:
            insights.append(f"🔁 Loyal listener - you replay your favorites more than {repeat_pct:.0f}% of users")
        
        # Danceability
        dance_pct = percentiles.get('avg_danceability', 50)
        if dance_pct > 75:
            insights.append(f"💃 Dance floor ready! Higher danceability than {dance_pct:.0f}% of listeners")
        
        # Valence (happiness)
        valence_pct = percentiles.get('avg_valence', 50)
        if valence_pct > 70:
            insights.append(f"😊 Happy vibes! Your music is more upbeat than {valence_pct:.0f}% of users")
        elif valence_pct < 30:
            insights.append(f"🎭 You embrace the feels - more emotional depth in your taste")
        
        return insights
    
    def generate_comparison_stats(self, user_features: Dict) -> Dict:
        """
        Generate comparison statistics for wrapped display.
        
        Args:
            user_features: User feature dictionary
            
        Returns:
            Dictionary of comparison metrics
        """
        percentiles = self.calculate_percentiles(user_features)
        classification = self.classify_user(user_features)
        insights = self.get_listening_insights(user_features, percentiles)
        
        # Key comparison metrics
        comparisons = []
        
        # Energy comparison
        energy_pct = percentiles.get('avg_energy', 50)
        comparisons.append({
            'metric': 'Energy Level',
            'value': user_features.get('avg_energy', 0.5),
            'percentile': energy_pct,
            'label': f"Top {100-energy_pct:.0f}%" if energy_pct > 50 else f"Bottom {energy_pct:.0f}%"
        })
        
        # Genre diversity
        diversity_pct = percentiles.get('genre_diversity', 50)
        comparisons.append({
            'metric': 'Genre Explorer',
            'value': user_features.get('genre_diversity', 0),
            'percentile': diversity_pct,
            'label': f"Top {100-diversity_pct:.0f}%" if diversity_pct > 50 else f"Bottom {diversity_pct:.0f}%"
        })
        
        # Night listener
        night_pct = percentiles.get('night_listener_ratio', 50)
        comparisons.append({
            'metric': 'Night Owl Score',
            'value': user_features.get('night_listener_ratio', 0),
            'percentile': night_pct,
            'label': f"Top {100-night_pct:.0f}%" if night_pct > 50 else f"Bottom {night_pct:.0f}%"
        })
        
        # Replay ratio
        repeat_pct = percentiles.get('repeat_artist_ratio', 50)
        comparisons.append({
            'metric': 'Replay Ratio',
            'value': user_features.get('repeat_artist_ratio', 0),
            'percentile': repeat_pct,
            'label': f"Top {100-repeat_pct:.0f}%" if repeat_pct > 50 else f"Bottom {repeat_pct:.0f}%"
        })
        
        return {
            'user_type': classification,
            'percentiles': percentiles,
            'insights': insights,
            'comparisons': comparisons
        }
    
    def save_model(self, filepath: str) -> None:
        """Save classifier to disk."""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'cluster_labels': self.cluster_labels,
            'fitted': self.fitted
        }, filepath)
        logger.info(f"User classifier saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load classifier from disk."""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.cluster_labels = data['cluster_labels']
        self.fitted = data['fitted']
        logger.info(f"User classifier loaded from {filepath}")


# Usage Example
if __name__ == "__main__":
    # Sample user features
    sample_user = {
        'genre_diversity': 0.05,
        'repeat_artist_ratio': 0.35,
        'night_listener_ratio': 0.25,
        'avg_energy': 0.72,
        'avg_valence': 0.58,
        'avg_danceability': 0.68,
        'avg_acousticness': 0.22,
        'explicit_ratio': 0.18,
        'total_tracks': 2400,
        'total_artists': 450
    }
    
    # Initialize classifier
    classifier = UserClassifier()
    
    # Classify user
    result = classifier.classify_user(sample_user)
    print(f"\n🎵 USER TYPE: {result['emoji']} {result['type_name']}")
    print(f"   {result['description']}")
    print(f"   Traits: {', '.join(result['traits'])}")
    print(f"   Confidence: {result['confidence']*100:.1f}%")
    
    # Get percentiles
    percentiles = classifier.calculate_percentiles(sample_user)
    print(f"\n📊 PERCENTILE RANKINGS:")
    for feature, pct in percentiles.items():
        print(f"   {feature}: Top {100-pct:.1f}%" if pct > 50 else f"   {feature}: {pct:.1f}th percentile")
    
    # Get insights
    insights = classifier.get_listening_insights(sample_user, percentiles)
    print(f"\n💡 INSIGHTS:")
    for insight in insights:
        print(f"   {insight}")
    
    # Full comparison stats
    comparison = classifier.generate_comparison_stats(sample_user)
    print(f"\n📈 COMPARISONS:")
    for comp in comparison['comparisons']:
        print(f"   {comp['metric']}: {comp['label']}")
