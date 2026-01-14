"""
FastAPI Backend - Spotify Wrapped AI/ML Workshop
Complete REST API for AI/ML models and analytics
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Import ML models (in production, these would be imported from app.ml module)
# from app.ml.taste_matcher import TasteMatcher
# from app.ml.mood_analyzer import MoodAnalyzer
# from app.ml.genre_classifier import GenreClassifier
# from app.ml.hybrid_recommender import HybridRecommender

logger = logging.getLogger(__name__)

# ============================================================================
# PYDANTIC SCHEMAS
# ============================================================================

class AudioFeaturesInput(BaseModel):
    track_uri: str
    track_name: str
    energy: float
    valence: float
    danceability: float
    tempo: float
    acousticness: float
    loudness: float
    speechiness: float
    instrumentalness: float
    liveness: float
    mode: int
    key: int
    time_signature: int
    duration_ms: int
    explicit: bool
    popularity: int
    genres: List[str] = []
    artist_names: str
    album_name: str
    release_date: str


class InteractionInput(BaseModel):
    user_id: str
    track_id: str
    interaction_type: str  # 'listen', 'save', 'replay', 'skip'
    duration_ms: int  # Time listened (for skip detection)
    timestamp: datetime


class TasteProfileResponse(BaseModel):
    user_id: str
    taste_match_scores: Dict[str, float]
    similar_users: List[Dict[str, float]]
    top_matching_tracks: List[Dict]


class MoodAnalysisResponse(BaseModel):
    user_id: str
    dominant_mood: str
    mood_distribution: Dict[str, float]
    insights: List[str]
    cluster_profiles: Dict


class GenreProfileResponse(BaseModel):
    user_id: str
    top_genres: List[str]
    genre_distribution: Dict[str, float]
    genre_probabilities: Dict


class UserTypeResponse(BaseModel):
    user_id: str
    user_type: str
    characteristics: Dict
    percentile_stats: Dict[str, float]


class ListeningPhaseResponse(BaseModel):
    phase_month: str
    phase_name: str
    characteristics: Dict
    tracks_count: int


class SpotifyWrappedResponse(BaseModel):
    user_id: str
    generated_at: datetime
    taste_profile: TasteProfileResponse
    mood_analysis: MoodAnalysisResponse
    genre_profile: GenreProfileResponse
    user_type: UserTypeResponse
    listening_phases: List[ListeningPhaseResponse]
    recommendations: List[Dict]
    stats: Dict


class RecommendationResponse(BaseModel):
    track_id: str
    track_name: str
    artist_names: str
    album_name: str
    final_score: float
    score_breakdown: Dict[str, float]
    reason: str


# ============================================================================
# FASTAPI APP SETUP
# ============================================================================

app = FastAPI(
    title="Spotify Wrapped AI/ML API",
    description="Complete AI/ML modeling + API for Spotify music analysis",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instances (in production, load from saved files)
taste_matcher = None
mood_analyzer = None
genre_classifier = None
recommender = None


# ============================================================================
# INITIALIZATION
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load trained models on startup."""
    global taste_matcher, mood_analyzer, genre_classifier, recommender
    
    logger.info("Loading trained ML models...")
    
    try:
        # Load models from saved files
        # taste_matcher = TasteMatcher()
        # taste_matcher.load_model("models/taste_matcher.pkl")
        # 
        # mood_analyzer = MoodAnalyzer()
        # mood_analyzer.load_model("models/mood_analyzer.pkl")
        # 
        # genre_classifier = GenreClassifier()
        # genre_classifier.load_model("models/genre_classifier.pkl")
        # 
        # recommender = HybridRecommender(
        #     taste_matcher, mood_analyzer, genre_classifier
        # )
        
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "models_loaded": taste_matcher is not None
    }


# ============================================================================
# TASTE MATCHING ENDPOINTS
# ============================================================================

@app.post("/api/v1/taste-profile", response_model=TasteProfileResponse)
async def get_taste_profile(user_id: str):
    """
    Get comprehensive taste profile for user.
    
    Returns:
    - User embeddings summary
    - Taste match scores with known tracks
    - Similar users
    - Top matching tracks
    """
    if taste_matcher is None:
        raise HTTPException(status_code=500, detail="Taste model not loaded")
    
    try:
        # Get user taste profile
        profile = taste_matcher.get_user_taste_profile(user_id)
        
        # Get similar users
        similar_users = taste_matcher.get_similar_users(user_id, n=10)
        
        return TasteProfileResponse(
            user_id=user_id,
            taste_match_scores=profile,
            similar_users=[{"user_id": u, "similarity": s} for u, s in similar_users],
            top_matching_tracks=[]  # Would populate from database
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ============================================================================
# MOOD ANALYSIS ENDPOINTS
# ============================================================================

@app.post("/api/v1/mood-analysis", response_model=MoodAnalysisResponse)
async def get_mood_analysis(user_id: str, user_tracks_data: List[AudioFeaturesInput]):
    """
    Analyze user's mood distribution from listening history.
    
    Returns:
    - Dominant mood/vibe
    - Distribution across mood clusters
    - Actionable insights
    """
    if mood_analyzer is None:
        raise HTTPException(status_code=500, detail="Mood model not loaded")
    
    try:
        # Convert to DataFrame
        tracks_df = pd.DataFrame([t.dict() for t in user_tracks_data])
        
        # Get mood analysis
        mood_insights = mood_analyzer.get_mood_insights(tracks_df)
        
        return MoodAnalysisResponse(
            user_id=user_id,
            dominant_mood=mood_insights['dominant_mood'],
            mood_distribution=mood_insights['full_distribution'],
            insights=mood_insights['insights'],
            cluster_profiles=mood_insights['cluster_profiles']
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# GENRE CLASSIFICATION ENDPOINTS
# ============================================================================

@app.post("/api/v1/genre-profile", response_model=GenreProfileResponse)
async def get_genre_profile(user_id: str, user_tracks_data: List[AudioFeaturesInput]):
    """
    Get user's meaningful genre profile.
    
    Returns:
    - Top genres (weighted by listening)
    - Genre distribution
    - Per-track genre probabilities
    """
    if genre_classifier is None:
        raise HTTPException(status_code=500, detail="Genre model not loaded")
    
    try:
        # Convert to DataFrame
        tracks_df = pd.DataFrame([t.dict() for t in user_tracks_data])
        
        # Get genre analysis
        weighted_genres = genre_classifier.get_weighted_genre_vector(tracks_df)
        top_genres = list(weighted_genres.keys())[:10]
        
        return GenreProfileResponse(
            user_id=user_id,
            top_genres=top_genres,
            genre_distribution=weighted_genres,
            genre_probabilities=weighted_genres
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# RECOMMENDATIONS ENDPOINT
# ============================================================================

@app.post("/api/v1/recommendations", response_model=List[RecommendationResponse])
async def get_recommendations(
    user_id: str,
    user_tracks_data: List[AudioFeaturesInput],
    candidate_tracks_data: List[AudioFeaturesInput],
    n_recommendations: int = 10
):
    """
    Get personalized track recommendations using hybrid engine.
    
    Scoring:
    - 50% Collaborative Filtering (taste match)
    - 30% Content-Based (mood + genre)
    - 20% Recent Taste Similarity
    
    Returns:
    - Top-N recommended tracks
    - Scores and explanations
    """
    if recommender is None:
        raise HTTPException(status_code=500, detail="Recommender not loaded")
    
    try:
        # Convert to DataFrames
        user_df = pd.DataFrame([t.dict() for t in user_tracks_data])
        candidates_df = pd.DataFrame([t.dict() for t in candidate_tracks_data])
        
        # Get recommendations
        recs = recommender.recommend_tracks(
            user_id, user_df, candidates_df,
            n_recommendations=n_recommendations
        )
        
        # Format response
        return [
            RecommendationResponse(
                track_id=rec['track_id'],
                track_name=rec['track_name'],
                artist_names=rec['artist_names'],
                album_name=rec['album_name'],
                final_score=rec['score'],
                score_breakdown=rec['score_breakdown'],
                reason=rec['reason']
            )
            for rec in recs
        ]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# SPOTIFY WRAPPED ENDPOINT
# ============================================================================

@app.post("/api/v1/spotify-wrapped", response_model=SpotifyWrappedResponse)
async def generate_spotify_wrapped(
    user_id: str,
    user_tracks_data: List[AudioFeaturesInput],
    candidate_tracks_data: Optional[List[AudioFeaturesInput]] = None
):
    """
    Generate complete Spotify Wrapped report with all analyses.
    
    Includes:
    1. Taste Profile (ALS embeddings)
    2. Mood Analysis (K-means clustering)
    3. Genre Profile (Random Forest classification)
    4. User Type Classification
    5. Listening Phases (Time evolution)
    6. Personalized Recommendations
    7. Stats & Comparisons
    """
    if taste_matcher is None or mood_analyzer is None or genre_classifier is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    try:
        # Convert to DataFrame
        user_df = pd.DataFrame([t.dict() for t in user_tracks_data])
        
        # 1. Taste Profile
        taste_profile = TasteProfileResponse(
            user_id=user_id,
            taste_match_scores={},
            similar_users=[],
            top_matching_tracks=[]
        )
        
        # 2. Mood Analysis
        mood_insights = mood_analyzer.get_mood_insights(user_df)
        mood_analysis = MoodAnalysisResponse(
            user_id=user_id,
            dominant_mood=mood_insights['dominant_mood'],
            mood_distribution=mood_insights['full_distribution'],
            insights=mood_insights['insights'],
            cluster_profiles=mood_insights['cluster_profiles']
        )
        
        # 3. Genre Profile
        weighted_genres = genre_classifier.get_weighted_genre_vector(user_df)
        genre_profile = GenreProfileResponse(
            user_id=user_id,
            top_genres=list(weighted_genres.keys())[:10],
            genre_distribution=weighted_genres,
            genre_probabilities=weighted_genres
        )
        
        # 4. User Type (would implement similar logic)
        user_type = UserTypeResponse(
            user_id=user_id,
            user_type="Music Explorer",
            characteristics={},
            percentile_stats={}
        )
        
        # 5. Listening Phases (would extract from temporal data)
        listening_phases = []
        
        # 6. Recommendations
        recommendations = []
        if candidate_tracks_data:
            candidates_df = pd.DataFrame([t.dict() for t in candidate_tracks_data])
            recs = recommender.recommend_tracks(
                user_id, user_df, candidates_df, n_recommendations=10
            )
            recommendations = recs
        
        # 7. Stats
        stats = {
            'total_tracks': len(user_df),
            'avg_energy': float(user_df['energy'].mean()),
            'avg_valence': float(user_df['valence'].mean()),
            'avg_tempo': float(user_df['tempo'].mean()),
            'most_common_key': int(user_df['key'].mode()[0]) if len(user_df) > 0 else 0,
            'explicit_percentage': float((user_df['explicit'].sum() / len(user_df) * 100))
        }
        
        return SpotifyWrappedResponse(
            user_id=user_id,
            generated_at=datetime.now(),
            taste_profile=taste_profile,
            mood_analysis=mood_analysis,
            genre_profile=genre_profile,
            user_type=user_type,
            listening_phases=listening_phases,
            recommendations=recommendations,
            stats=stats
        )
    except Exception as e:
        logger.error(f"Error generating Spotify Wrapped: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# TRACK UPLOAD ENDPOINT
# ============================================================================

@app.post("/api/v1/import-tracks")
async def import_tracks(tracks: List[AudioFeaturesInput]):
    """
    Import tracks into the system for analysis.
    
    This endpoint receives track data with all audio features
    and stores them for model training and inference.
    """
    try:
        # In production: Save to database
        # db.add_tracks(tracks)
        
        return {
            "status": "success",
            "tracks_imported": len(tracks),
            "timestamp": datetime.now()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# INTERACTION LOGGING ENDPOINT
# ============================================================================

@app.post("/api/v1/log-interaction")
async def log_interaction(interaction: InteractionInput):
    """
    Log user interaction for implicit feedback.
    
    Types:
    - listen: Full track played
    - save: User saved/liked track
    - replay: User replayed track
    - skip: User skipped track before 30s
    """
    try:
        # In production: Save to database
        # db.add_interaction(interaction)
        
        return {
            "status": "success",
            "interaction_logged": True,
            "timestamp": datetime.now()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# MODEL TRAINING ENDPOINT (Admin Only)
# ============================================================================

@app.post("/api/v1/train-models")
async def train_models(background_tasks: BackgroundTasks):
    """
    Trigger training of all ML models (background task).
    
    In production:
    - Load interaction data from database
    - Train ALS, K-Means, Random Forest
    - Save models
    """
    def train_all_models():
        logger.info("Starting model training...")
        # Training logic here
        logger.info("Model training complete")
    
    background_tasks.add_task(train_all_models)
    
    return {
        "status": "training_started",
        "message": "Models are being trained in the background"
    }


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
