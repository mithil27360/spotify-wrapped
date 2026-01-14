"""
DATA PIPELINE - Load and Process Spotify Listening Data
Handles CSV parsing, feature engineering, and data preparation for ML models
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SpotifyDataPipeline:
    """
    Comprehensive data pipeline for Spotify listening data.
    
    Features:
    - CSV loading with proper parsing
    - Genre string → list conversion
    - Time feature extraction
    - Interaction weight calculation
    - User feature aggregation
    """
    
    # Audio feature columns
    AUDIO_FEATURES = [
        'Danceability', 'Energy', 'Key', 'Loudness', 'Mode',
        'Speechiness', 'Acousticness', 'Instrumentalness',
        'Liveness', 'Valence', 'Tempo', 'Time Signature'
    ]
    
    # Normalized column names
    AUDIO_FEATURES_NORMALIZED = [
        'danceability', 'energy', 'key', 'loudness', 'mode',
        'speechiness', 'acousticness', 'instrumentalness',
        'liveness', 'valence', 'tempo', 'time_signature'
    ]
    
    def __init__(self, csv_path: str = "data/SytheticData1000.csv"):
        """
        Initialize data pipeline.
        
        Args:
            csv_path: Path to the Liked_Songs.csv file
        """
        self.csv_path = csv_path
        self.raw_df = None
        self.processed_df = None
        self.user_features = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load and parse the CSV file.
        
        Returns:
            Raw DataFrame with parsed columns
        """
        logger.info(f"Loading data from {self.csv_path}")
        
        self.raw_df = pd.read_csv(self.csv_path)
        
        # Normalize column names to lowercase with underscores
        self.raw_df.columns = [col.strip().lower().replace(' ', '_') for col in self.raw_df.columns]
        
        logger.info(f"Loaded {len(self.raw_df)} tracks")
        return self.raw_df
    
    def parse_genres(self, genres_str: str) -> List[str]:
        """
        Parse genre string into list.
        
        Args:
            genres_str: Comma-separated genre string
            
        Returns:
            List of genre strings
        """
        if pd.isna(genres_str) or genres_str == '':
            return []
        return [g.strip() for g in str(genres_str).split(',')]
    
    def extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract time-based features from 'added_at' column.
        
        Args:
            df: DataFrame with 'added_at' column
            
        Returns:
            DataFrame with added time features
        """
        df = df.copy()
        
        # Parse datetime
        df['added_at'] = pd.to_datetime(df['added_at'], errors='coerce')
        
        # Extract time features
        df['added_hour'] = df['added_at'].dt.hour
        df['added_day'] = df['added_at'].dt.dayofweek  # 0=Monday
        df['added_month'] = df['added_at'].dt.month
        df['added_year'] = df['added_at'].dt.year
        df['added_date'] = df['added_at'].dt.date
        
        # Is weekend (Saturday=5, Sunday=6)
        df['is_weekend'] = df['added_day'].isin([5, 6])
        
        # Is night (10PM - 6AM)
        df['is_night'] = df['added_hour'].apply(
            lambda h: h >= 22 or h < 6 if pd.notna(h) else False
        )
        
        return df
    
    def process_data(self) -> pd.DataFrame:
        """
        Full data processing pipeline.
        
        Returns:
            Processed DataFrame ready for analysis
        """
        if self.raw_df is None:
            self.load_data()
        
        df = self.raw_df.copy()
        
        # Parse genres
        df['genres_list'] = df['genres'].apply(self.parse_genres)
        df['primary_genre'] = df['genres_list'].apply(
            lambda x: x[0] if len(x) > 0 else 'unknown'
        )
        df['genre_count'] = df['genres_list'].apply(len)
        
        # Extract time features
        df = self.extract_time_features(df)
        
        # Parse release date
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df['release_year'] = df['release_date'].dt.year
        
        # Duration in minutes - handle both column naming conventions
        if 'duration_(ms)' in df.columns:
            df['duration_min'] = df['duration_(ms)'] / 60000
        elif 'duration_ms' in df.columns:
            df['duration_min'] = df['duration_ms'] / 60000
        else:
            df['duration_min'] = 3.5  # default 3.5 min
        
        # Normalize boolean
        df['explicit'] = df['explicit'].astype(bool)
        
        # Fill missing audio features with median
        for col in self.AUDIO_FEATURES_NORMALIZED:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        self.processed_df = df
        logger.info(f"Processed {len(df)} tracks with {len(df.columns)} features")
        
        return df
    
    def get_audio_features_df(self) -> pd.DataFrame:
        """
        Get DataFrame with just audio features (normalized).
        
        Returns:
            DataFrame with audio features
        """
        if self.processed_df is None:
            self.process_data()
        
        # Available audio feature columns
        available_cols = [col for col in self.AUDIO_FEATURES_NORMALIZED 
                         if col in self.processed_df.columns]
        
        return self.processed_df[available_cols].copy()
    
    def calculate_user_features(self) -> Dict:
        """
        Calculate aggregate user features for classification.
        
        Returns:
            Dictionary of user features
        """
        if self.processed_df is None:
            self.process_data()
        
        df = self.processed_df
        
        # Audio feature averages
        avg_features = {}
        for col in self.AUDIO_FEATURES_NORMALIZED:
            if col in df.columns:
                avg_features[f'avg_{col}'] = float(df[col].mean())
        
        # Genre diversity
        all_genres = [g for genres in df['genres_list'] for g in genres]
        unique_genres = set(all_genres)
        genre_counts = pd.Series(all_genres).value_counts()
        
        # Time patterns
        night_ratio = df['is_night'].mean() if 'is_night' in df.columns else 0
        weekend_ratio = df['is_weekend'].mean() if 'is_weekend' in df.columns else 0
        
        # Repeat detection (same artist appearing multiple times)
        artist_counts = df['artist_name(s)'].value_counts()
        repeat_ratio = (artist_counts > 1).sum() / len(artist_counts) if len(artist_counts) > 0 else 0
        
        # Explicit content ratio
        explicit_ratio = df['explicit'].mean() if 'explicit' in df.columns else 0
        
        self.user_features = {
            # Audio averages
            **avg_features,
            
            # Genre metrics
            'total_genres': len(unique_genres),
            'genre_diversity': len(unique_genres) / len(all_genres) if len(all_genres) > 0 else 0,
            'top_genre': genre_counts.index[0] if len(genre_counts) > 0 else 'unknown',
            'top_genre_percentage': (genre_counts.iloc[0] / len(all_genres) * 100) 
                                    if len(genre_counts) > 0 else 0,
            
            # Time patterns
            'night_listener_ratio': float(night_ratio),
            'weekend_ratio': float(weekend_ratio),
            
            # Behavior
            'repeat_artist_ratio': float(repeat_ratio),
            'explicit_ratio': float(explicit_ratio),
            
            # Stats
            'total_tracks': len(df),
            'total_artists': df['artist_name(s)'].nunique(),
            'avg_popularity': float(df['popularity'].mean()) if 'popularity' in df.columns else 0,
            'avg_duration_min': float(df['duration_min'].mean()) if 'duration_min' in df.columns else 0,
        }
        
        return self.user_features
    
    def get_monthly_taste_vectors(self) -> pd.DataFrame:
        """
        Build monthly taste vectors for phase analysis.
        
        Returns:
            DataFrame with monthly aggregate features
        """
        if self.processed_df is None:
            self.process_data()
        
        df = self.processed_df.copy()
        
        # Create year-month column
        df['year_month'] = df['added_at'].dt.to_period('M')
        
        # Audio features to aggregate
        agg_cols = ['energy', 'valence', 'danceability', 'tempo', 'acousticness']
        available_cols = [col for col in agg_cols if col in df.columns]
        
        # Group by month
        monthly = df.groupby('year_month').agg({
            **{col: 'mean' for col in available_cols},
            'track_uri': 'count'
        }).rename(columns={'track_uri': 'track_count'})
        
        # Get dominant genre per month
        def get_dominant_genre(group):
            all_genres = [g for genres in group['genres_list'] for g in genres]
            if len(all_genres) == 0:
                return 'unknown'
            return pd.Series(all_genres).value_counts().index[0]
        
        monthly['dominant_genre'] = df.groupby('year_month').apply(get_dominant_genre)
        
        return monthly.reset_index()
    
    def get_top_artists(self, n: int = 10) -> pd.DataFrame:
        """
        Get top N artists by track count.
        
        Args:
            n: Number of top artists
            
        Returns:
            DataFrame with artist stats
        """
        if self.processed_df is None:
            self.process_data()
        
        df = self.processed_df
        
        artist_stats = df.groupby('artist_name(s)').agg({
            'track_uri': 'count',
            'popularity': 'mean',
            'energy': 'mean',
            'valence': 'mean'
        }).rename(columns={
            'track_uri': 'track_count',
            'popularity': 'avg_popularity',
            'energy': 'avg_energy',
            'valence': 'avg_valence'
        }).sort_values('track_count', ascending=False)
        
        return artist_stats.head(n).reset_index()
    
    def get_top_tracks(self, n: int = 10) -> pd.DataFrame:
        """
        Get top N tracks by popularity.
        
        Args:
            n: Number of top tracks
            
        Returns:
            DataFrame with track info
        """
        if self.processed_df is None:
            self.process_data()
        
        cols = ['track_name', 'artist_name(s)', 'album_name', 'popularity', 
                'energy', 'valence', 'danceability', 'primary_genre']
        available_cols = [col for col in cols if col in self.processed_df.columns]
        
        return self.processed_df.nlargest(n, 'popularity')[available_cols].reset_index(drop=True)
    
    def get_genre_distribution(self) -> Dict[str, float]:
        """
        Get genre distribution (percentage of tracks).
        
        Returns:
            Dict mapping genre → percentage
        """
        if self.processed_df is None:
            self.process_data()
        
        all_genres = [g for genres in self.processed_df['genres_list'] for g in genres]
        genre_counts = pd.Series(all_genres).value_counts()
        
        total = len(all_genres)
        return {genre: count / total * 100 for genre, count in genre_counts.head(20).items()}
    
    def get_listening_stats(self) -> Dict:
        """
        Get comprehensive listening statistics.
        
        Returns:
            Dictionary of listening stats
        """
        if self.processed_df is None:
            self.process_data()
        
        df = self.processed_df
        
        # Total listening time (estimate: each track = 1 play)
        # Handle both column naming conventions
        if 'duration_(ms)' in df.columns:
            total_duration_ms = df['duration_(ms)'].sum()
        elif 'duration_ms' in df.columns:
            total_duration_ms = df['duration_ms'].sum()
        else:
            total_duration_ms = len(df) * 3.5 * 60000  # default 3.5 min per track
        total_hours = total_duration_ms / (1000 * 60 * 60)
        
        # Date range
        min_date = df['added_at'].min()
        max_date = df['added_at'].max()
        
        return {
            'total_tracks': len(df),
            'total_artists': df['artist_name(s)'].nunique(),
            'total_albums': df['album_name'].nunique(),
            'total_hours': round(total_hours, 1),
            'total_minutes': round(total_duration_ms / (1000 * 60), 0),
            'avg_track_duration_min': round(df['duration_min'].mean(), 2),
            'date_range_start': min_date.strftime('%Y-%m-%d') if pd.notna(min_date) else 'N/A',
            'date_range_end': max_date.strftime('%Y-%m-%d') if pd.notna(max_date) else 'N/A',
            'explicit_tracks': int(df['explicit'].sum()),
            'explicit_percentage': round(df['explicit'].mean() * 100, 1),
            'avg_popularity': round(df['popularity'].mean(), 1),
        }


# Usage Example
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = SpotifyDataPipeline("data/SytheticData1000.csv")
    
    # Load and process data
    df = pipeline.process_data()
    print(f"\n📊 Loaded {len(df)} tracks")
    print(f"Columns: {list(df.columns)[:10]}...")
    
    # Get listening stats
    stats = pipeline.get_listening_stats()
    print(f"\n🎵 Listening Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Get user features
    user_features = pipeline.calculate_user_features()
    print(f"\n👤 User Features:")
    for key, value in list(user_features.items())[:10]:
        print(f"  {key}: {value}")
    
    # Top artists
    print(f"\n🎤 Top 5 Artists:")
    top_artists = pipeline.get_top_artists(5)
    for _, row in top_artists.iterrows():
        print(f"  {row['artist_name(s)']}: {row['track_count']} tracks")
    
    # Genre distribution
    print(f"\n🎸 Top 5 Genres:")
    genres = pipeline.get_genre_distribution()
    for genre, pct in list(genres.items())[:5]:
        print(f"  {genre}: {pct:.1f}%")
