"""
HYBRID RECOMMENDATION ENGINE - Collaborative Filtering + Content-Based
Scoring: 0.5*CF + 0.3*mood/genre + 0.2*recent_taste
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class HybridRecommender:
    """
    Hybrid recommendation system combining:
    1. Collaborative Filtering (user-track embeddings)
    2. Genre & Mood Matching
    3. Recent Taste Similarity
    """
    
    def __init__(self, taste_matcher, mood_analyzer, genre_classifier):
        """
        Initialize recommender with trained ML models.
        
        Args:
            taste_matcher: Trained TasteMatcher (ALS)
            mood_analyzer: Trained MoodAnalyzer (K-means)
            genre_classifier: Trained GenreClassifier (Random Forest)
        """
        self.taste_matcher = taste_matcher
        self.mood_analyzer = mood_analyzer
        self.genre_classifier = genre_classifier
    
    def get_collaborative_score(self, user_id: str, track_id: str) -> float:
        """
        Get collaborative filtering score (user-track embedding similarity).
        
        Range: 0 to 1 (normalized)
        """
        try:
            score = self.taste_matcher.get_taste_match_score(user_id, track_id)
            return score
        except ValueError:
            return 0.5  # Default neutral score
    
    def get_mood_match_score(self, user_df: pd.DataFrame, 
                             track_row: pd.Series) -> float:
        """
        Get mood/vibe matching score.
        
        Returns:
        - 1.0 if track's mood matches user's dominant mood perfectly
        - Cosine similarity of mood distributions otherwise
        """
        # Get user's mood distribution
        user_mood_dist = self.mood_analyzer.get_user_mood_distribution(user_df)
        
        # Get track's mood (which cluster does it fall into)
        track_features = track_row[['energy', 'valence', 'danceability', 
                                     'tempo', 'acousticness']].to_frame().T
        
        # Predict track's cluster
        feature_scaled = self.mood_analyzer.scaler.transform(track_features)
        track_cluster = self.mood_analyzer.kmeans.predict(feature_scaled)[0]
        track_mood = self.mood_analyzer.cluster_profiles[track_cluster]['label']
        
        # Calculate similarity
        if track_mood in user_mood_dist:
            mood_match = user_mood_dist[track_mood] / 100.0
        else:
            mood_match = 0.2  # Small bonus if not in user's main moods
        
        return min(1.0, mood_match)
    
    def get_genre_match_score(self, user_df: pd.DataFrame, 
                              track_row: pd.Series,
                              user_weights: pd.Series = None) -> float:
        """
        Get genre matching score.
        
        Returns:
        - Cosine similarity between user's genre vector and track's genre vector
        """
        # Get user's weighted genre vector
        user_genres = self.genre_classifier.get_weighted_genre_vector(
            user_df, user_weights
        )
        user_genre_vector = np.array(list(user_genres.values()))
        
        # Get track's genre vector
        track_df = pd.DataFrame([track_row])
        track_genres = self.genre_classifier.predict_genre_probabilities(track_df).iloc[0]
        track_genre_vector = track_genres.values
        
        # Cosine similarity
        if len(user_genre_vector) > 0 and len(track_genre_vector) > 0:
            similarity = cosine_similarity(
                user_genre_vector.reshape(1, -1),
                track_genre_vector.reshape(1, -1)
            )[0, 0]
            return float(max(0, similarity))  # Clamp to [0, 1]
        
        return 0.5
    
    def get_recent_taste_score(self, user_df: pd.DataFrame, 
                              track_row: pd.Series,
                              recent_days: int = 30) -> float:
        """
        Get similarity between track and user's recent listening.
        
        Requires 'added_at' column in user_df to filter recent tracks.
        """
        # Filter recent tracks if date column exists
        if 'added_at' in user_df.columns:
            recent_df = user_df[
                (pd.to_datetime(user_df['added_at']) >= 
                 pd.Timestamp.now() - pd.Timedelta(days=recent_days))
            ]
            if len(recent_df) == 0:
                recent_df = user_df  # Fallback to all if no recent
        else:
            recent_df = user_df
        
        # Get recent taste vector (average audio features)
        audio_features = ['energy', 'valence', 'danceability', 'tempo', 'acousticness']
        recent_vector = recent_df[audio_features].mean().values
        track_vector = track_row[audio_features].values
        
        # Cosine similarity
        similarity = cosine_similarity(
            recent_vector.reshape(1, -1),
            track_vector.reshape(1, -1)
        )[0, 0]
        
        return float(max(0, similarity))
    
    def get_hybrid_score(self, user_id: str, user_df: pd.DataFrame, 
                        track_row: pd.Series, track_id: str,
                        user_weights: pd.Series = None) -> Dict[str, float]:
        """
        Get hybrid recommendation score combining all signals.
        
        Weighting:
        - 0.5: Collaborative Filtering
        - 0.3: Genre & Mood Match (equally weighted)
        - 0.2: Recent Taste Similarity
        
        Returns:
            Dict with component scores and final score
        """
        # Get component scores
        cf_score = self.get_collaborative_score(user_id, track_id)
        mood_score = self.get_mood_match_score(user_df, track_row)
        genre_score = self.get_genre_match_score(user_df, track_row, user_weights)
        recent_score = self.get_recent_taste_score(user_df, track_row)
        
        # Content-based composite (mood + genre equally weighted)
        content_score = (mood_score + genre_score) / 2.0
        
        # Final hybrid score
        final_score = (
            0.5 * cf_score +      # Collaborative filtering
            0.3 * content_score +  # Content-based (mood + genre)
            0.2 * recent_score     # Recent taste
        )
        
        return {
            'collaborative_filtering': float(cf_score),
            'mood_match': float(mood_score),
            'genre_match': float(genre_score),
            'recent_taste': float(recent_score),
            'content_based': float(content_score),
            'final_score': float(final_score),
            'explanation': {
                'cf': f"User-track taste match: {cf_score:.2%}",
                'mood': f"Mood compatibility: {mood_score:.2%}",
                'genre': f"Genre overlap: {genre_score:.2%}",
                'recent': f"Recent listening match: {recent_score:.2%}"
            }
        }
    
    def recommend_tracks(self, user_id: str, user_df: pd.DataFrame,
                        candidate_tracks: pd.DataFrame,
                        n_recommendations: int = 10,
                        user_weights: pd.Series = None) -> List[Dict]:
        """
        Get personalized track recommendations.
        
        Args:
            user_id: User identifier
            user_df: User's listening history
            candidate_tracks: Pool of candidate tracks
            n_recommendations: Number of tracks to recommend
            user_weights: Optional weights for user's tracks
            
        Returns:
            List of recommended tracks with scores and explanations
        """
        recommendations = []
        
        for idx, (_, track_row) in enumerate(candidate_tracks.iterrows()):
            track_id = track_row.get('track_id', f'track_{idx}')
            
            # Skip if already in user's library (simple check)
            if 'track_id' in user_df.columns and track_id in user_df['track_id'].values:
                continue
            
            # Get hybrid score
            score_breakdown = self.get_hybrid_score(
                user_id, user_df, track_row, track_id, user_weights
            )
            
            recommendations.append({
                'track_id': track_id,
                'track_name': track_row.get('track_name', 'Unknown'),
                'artist_names': track_row.get('artist_names', 'Unknown'),
                'album_name': track_row.get('album_name', 'Unknown'),
                'score': score_breakdown['final_score'],
                'score_breakdown': score_breakdown,
                'reason': self._generate_reason(score_breakdown['explanation'])
            })
        
        # Sort by score and get top N
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return recommendations[:n_recommendations]
    
    def _generate_reason(self, explanation: Dict[str, str]) -> str:
        """Generate human-readable recommendation reason."""
        reasons = []
        
        for key, text in explanation.items():
            if key != 'cf':
                reasons.append(text)
        
        return " | ".join(reasons[:2])  # Top 2 reasons


# Usage Example
if __name__ == "__main__":
    print("Hybrid Recommender initialized (requires trained models)")
    print("Usage: recommender.recommend_tracks(user_id, user_df, candidates, n=10)")
