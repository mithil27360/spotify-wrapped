"""
Spotify Wrapped Application
Streamlit-based interactive data storytelling application for analyzing Spotify listening history.
Features: Data pipeline processing, user classification, mood analysis, and recommender system integration.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import time
import sys
import os
import textwrap

# Add src and ml directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ml'))

# Import our modules
from data_pipeline import SpotifyDataPipeline


from user_classifier import UserClassifier
from mood_analyzer import MoodAnalyzer

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Your Spotify Wrapped",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# SPOTIFY WRAPPED CSS - AUTHENTIC STYLE
# ============================================================================

# Gradient presets for each slide
GRADIENTS = {
    'upload': 'linear-gradient(180deg, #0b1220 0%, #0a1a2f 100%)',
    'intro': 'linear-gradient(135deg, #5b2cff 0%, #ff3cac 100%)',
    'stats': 'linear-gradient(135deg, #00dbde 0%, #fc00ff 100%)',
    'artist': 'linear-gradient(135deg, #ff512f 0%, #f09819 100%)',
    'vibe': 'linear-gradient(135deg, #11998e 0%, #38ef7d 100%)',
    'genre': 'linear-gradient(135deg, #8E2DE2 0%, #4A00E0 100%)',
    'type': 'linear-gradient(135deg, #f857a6 0%, #ff5858 100%)',
    'compare': 'linear-gradient(135deg, #00c6ff 0%, #0072ff 100%)',
    'share': 'linear-gradient(135deg, #1DB954 0%, #1ed760 100%)',
}

def inject_css(gradient: str = GRADIENTS['upload']):
    """Inject CSS with specified gradient background."""
    st.markdown(f"""
    <style>
        /* Hide Streamlit UI */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}
        
        /* Full screen gradient */
        .stApp {{
            background: {gradient};
            color: white;
        }}
        
        .main .block-container {{
            padding: 1rem 2rem;
            max-width: 100%;
        }}
        
        /* Center everything */
        .center {{
            text-align: center;
        }}
        
        /* Primary Title Typography */
        .mega-title {{
            font-size: 6rem !important;
            font-weight: 900 !important;
            text-align: center !important;
            color: white !important;
            text-transform: uppercase;
            letter-spacing: -3px;
            line-height: 1 !important;
            margin: 0 !important;
            text-shadow: 4px 4px 8px rgba(0,0,0,0.3);
        }}
        
        .big-title {{
            font-size: 4rem !important;
            font-weight: 900 !important;
            text-align: center !important;
            color: white !important;
            line-height: 1.1 !important;
            margin: 0.5rem 0 !important;
        }}
        
        .subtitle {{
            font-size: 1.5rem !important;
            text-align: center !important;
            color: rgba(255,255,255,0.9) !important;
            font-weight: 500 !important;
        }}
        
        /* Green Accent Color */
        .green {{
            color: #1ed760 !important;
        }}
        
        /* Large Metric Display */
        .giant-number {{
            font-size: 10rem !important;
            font-weight: 900 !important;
            text-align: center !important;
            color: white !important;
            line-height: 1 !important;
            margin: 0 !important;
            text-shadow: 6px 6px 12px rgba(0,0,0,0.4);
        }}
        
        .number-label {{
            font-size: 2rem !important;
            text-align: center !important;
            color: rgba(255,255,255,0.9) !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 3px;
        }}
        
        /* Glassmorphism Container */
        .glass {{
            background: rgba(20, 60, 60, 0.4);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 25px;
            border: 1px solid rgba(255,255,255,0.15);
            padding: 2rem;
            margin: 1rem auto;
            max-width: 600px;
        }}
        
        /* Personality Type Container */
        .personality-card {{
            background: rgba(0,0,0,0.3);
            border-radius: 30px;
            padding: 3rem;
            text-align: center;
            margin: 2rem auto;
            max-width: 500px;
        }}
        
        .emoji-huge {{
            font-size: 8rem !important;
            margin: 0 !important;
        }}
        
        /* Progress bar */
        .progress-container {{
            background: rgba(255,255,255,0.2);
            border-radius: 15px;
            height: 20px;
            margin: 1rem 0;
            overflow: hidden;
        }}
        
        .progress-bar {{
            background: white;
            height: 100%;
            border-radius: 15px;
            transition: width 1s ease;
        }}
        
        /* Descriptive Copy */
        .fun-text {{
            font-size: 1.3rem !important;
            text-align: center !important;
            color: rgba(255,255,255,0.85) !important;
            font-style: italic;
            margin: 1rem 0 !important;
        }}
        
        /* Button styling */
        .stButton > button {{
            background: rgba(0,0,0,0.3) !important;
            color: white !important;
            font-weight: 700 !important;
            border: 2px solid rgba(255,255,255,0.3) !important;
            border-radius: 50px !important;
            padding: 1rem 3rem !important;
            font-size: 1.2rem !important;
            transition: all 0.3s ease !important;
        }}
        
        .stButton > button:hover {{
            background: rgba(255,255,255,0.2) !important;
            border-color: white !important;
            transform: scale(1.05);
        }}
        
        /* File uploader */
        [data-testid="stFileUploader"] {{
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 1rem;
        }}
        
        /* Hide upload label */
        [data-testid="stFileUploader"] label {{
            color: white !important;
        }}
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data
def load_wrapped_data_from_df(df_raw):
    """Load and process wrapped data from uploaded DataFrame."""
    try:
        pipeline = SpotifyDataPipeline.__new__(SpotifyDataPipeline)
        pipeline.csv_path = "uploaded"
        pipeline.raw_df = df_raw.copy()
        pipeline.processed_df = None
        pipeline.user_features = None
        pipeline.raw_df.columns = [col.strip().lower().replace(' ', '_') for col in pipeline.raw_df.columns]
        
        df = pipeline.process_data()
        stats = pipeline.get_listening_stats()
        user_features = pipeline.calculate_user_features()
        top_artists = pipeline.get_top_artists(10)
        genre_dist = pipeline.get_genre_distribution()
        monthly = pipeline.get_monthly_taste_vectors()
        
        classifier = UserClassifier()
        user_type = classifier.classify_user(user_features)
        comparison = classifier.generate_comparison_stats(user_features)
        
        # Fit Mood Analyzer on-the-fly
        mood_analyzer = MoodAnalyzer(n_clusters=4)
        mood_analyzer.fit(df)
        mood_insights = mood_analyzer.get_mood_insights(df)
        
        return {
            'df': df, 'stats': stats, 'user_features': user_features,
            'top_artists': top_artists, 'genre_dist': genre_dist,
            'monthly': monthly, 'user_type': user_type, 'comparison': comparison,
            'mood_analyzer': mood_analyzer, 'mood_insights': mood_insights
        }
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# ============================================================================
# ============================================================================
# VISUALIZATION COMPONENTS
# ============================================================================

def slide_upload():
    """Upload slide with instructions."""
    inject_css(GRADIENTS['upload'])
    
    st.markdown("""
        <div class="center" style="padding-top: 2rem;">
            <p class="green" style="font-size: 1.1rem; font-weight: 600;">
                Spotify Wrapped Workshop • Aurora ISTE MANIPAL
            </p>
            <p style="font-size: 1.8rem; color: rgba(255,255,255,0.7); margin: 0;">Your</p>
            <h1 class="mega-title">SPOTIFY WRAPPED</h1>
            <p class="subtitle green">🎵 Upload your data to begin</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Glass card with instructions
    st.markdown("""
        <div class="glass">
            <h3 class="green" style="text-align: center; margin-top: 0;">📋 How to Get Your Data</h3>
            <p style="font-size: 1.1rem; line-height: 2;"><strong>Step 1:</strong> Go to <a href="https://exportify.net/" target="_blank" style="color: #1ed760;">exportify.net</a></p>
            <p style="font-size: 1.1rem; line-height: 2;"><strong>Step 2:</strong> Click "Get Started" and connect Spotify</p>
            <p style="font-size: 1.1rem; line-height: 2;"><strong>Step 3:</strong> Find your "Liked Songs" playlist</p>
            <p style="font-size: 1.1rem; line-height: 2;"><strong>Step 4:</strong> Click "Export" to download CSV</p>
            <p style="font-size: 1.1rem; line-height: 2;"><strong>Step 5:</strong> Upload the file below 👇</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.link_button("🔗 Open Exportify.net", "https://exportify.net/", use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'], label_visibility="collapsed")
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.session_state.uploaded_df = df
            st.success(f"✅ Loaded {len(df):,} tracks!")
            if st.button("✨ Generate My Wrapped", use_container_width=True):
                st.session_state.slide = 1
                st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("📊 Try Sample Data", use_container_width=True):
            try:
                # Use path relative to this script's location
                script_dir = os.path.dirname(__file__)
                sample_path = os.path.join(script_dir, "data", "SytheticData1000.csv")
                df = pd.read_csv(sample_path)
                st.session_state.uploaded_df = df
                st.session_state.slide = 1
                st.rerun()
            except Exception as e:
                st.error(f"Sample data not found: {e}")
    
    st.markdown("""
        <p style="text-align: center; color: rgba(255,255,255,0.4); margin-top: 2rem; font-size: 0.9rem;">
            🔒 Your data stays private
        </p>
    """, unsafe_allow_html=True)


def slide_intro(data):
    """Intro slide - Let's go!"""
    inject_css(GRADIENTS['intro'])
    
    st.markdown("""
        <div class="center" style="padding-top: 15vh;">
            <p style="font-size: 2rem; margin: 0;">You're about to see</p>
            <h1 class="mega-title" style="font-size: 8rem;">YOUR</h1>
            <h1 class="mega-title" style="color: #1ed760;">WRAPPED</h1>
            <p class="fun-text" style="margin-top: 2rem;">Let's dive into your year in music 🎵</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        if st.button("Let's Go →", use_container_width=True):
            st.session_state.slide = 2
            st.rerun()


def slide_stats(data):
    """Stats slide - Big numbers."""
    inject_css(GRADIENTS['stats'])
    stats = data['stats']
    
    st.markdown(f"""
        <div class="center" style="padding-top: 5vh;">
            <p class="subtitle">This year, you listened to</p>
            <h1 class="giant-number">{stats['total_tracks']:,}</h1>
            <p class="number-label">Songs</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
            <div class="center">
                <p style="font-size: 4rem; font-weight: 900; margin: 0;">{stats['total_hours']:.0f}</p>
                <p class="subtitle">Hours</p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class="center">
                <p style="font-size: 4rem; font-weight: 900; margin: 0;">{stats['total_artists']:,}</p>
                <p class="subtitle">Artists</p>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
            <div class="center">
                <p style="font-size: 4rem; font-weight: 900; margin: 0;">{stats['total_albums']:,}</p>
                <p class="subtitle">Albums</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        <p class="fun-text" style="margin-top: 3rem;">That's a lot of music. You really lived it. 🎧</p>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Next →", use_container_width=True):
            st.session_state.slide = 3
            st.rerun()


def slide_top_artist(data):
    """Top artist slide - Hero moment."""
    inject_css(GRADIENTS['artist'])
    
    if len(data['top_artists']) == 0:
        st.warning("No artist data")
        return
    
    top = data['top_artists'].iloc[0]
    artist = top['artist_name(s)']
    count = top['track_count']
    
    st.markdown(f"""
        <div class="center" style="padding-top: 10vh;">
            <p class="subtitle">Your #1 Artist was</p>
            <h1 class="mega-title" style="font-size: 7rem;">{artist.upper()}</h1>
            <p class="number-label" style="margin-top: 2rem;">{count} songs in your library</p>
            <p class="fun-text">You were in your {artist.split()[0]} era ✨</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Top 5 list
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Your Top 5</p>", unsafe_allow_html=True)
    
    for i, (_, row) in enumerate(data['top_artists'].head(5).iterrows()):
        pct = row['track_count'] / data['top_artists'].head(5)['track_count'].max() * 100
        st.markdown(f"""
            <div style="max-width: 500px; margin: 0.5rem auto;">
                <div style="display: flex; justify-content: space-between;">
                    <span style="font-weight: 700;">#{i+1} {row['artist_name(s)']}</span>
                    <span style="color: rgba(255,255,255,0.8);">{row['track_count']} songs</span>
                </div>
                <div class="progress-container">
                    <div class="progress-bar" style="width: {pct}%;"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Next →", use_container_width=True):
            st.session_state.slide = 4
            st.rerun()


def slide_vibe(data):
    """Vibe/Mood slide."""
    inject_css(GRADIENTS['vibe'])
    uf = data['user_features']
    
    energy = uf.get('avg_energy', 0.5)
    valence = uf.get('avg_valence', 0.5)
    
    if energy > 0.6 and valence > 0.5:
        vibe = "HIGH ENERGY"
        vibe_sub = "You kept the party going all year 🔥"
    elif energy > 0.6:
        vibe = "INTENSE"
        vibe_sub = "Raw power defined your year ⚡"
    elif valence > 0.5:
        vibe = "GOOD VIBES"
        vibe_sub = "Positivity was your soundtrack 😊"
    else:
        vibe = "IN YOUR FEELS"
        vibe_sub = "You embraced the emotions 🌙"
    
    st.markdown(f"""
        <div class="center" style="padding-top: 10vh;">
            <p class="subtitle">Your vibe was</p>
            <h1 class="mega-title">{vibe}</h1>
            <p class="fun-text">{vibe_sub}</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Audio DNA - Radar Chart
    st.markdown("<br>", unsafe_allow_html=True)
    
    categories = ['Energy', 'Happiness', 'Danceability', 'Acousticness']
    r_values = [
        uf.get('avg_energy', 0.5),
        uf.get('avg_valence', 0.5),
        uf.get('avg_danceability', 0.5),
        uf.get('avg_acousticness', 0.5)
    ]
    
    # Add first point to end to close the polygon
    r_values_closed = r_values + [r_values[0]]
    categories_closed = categories + [categories[0]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=r_values_closed,
        theta=categories_closed,
        fill='toself',
        name='You',
        line=dict(color='#1DB954', width=3),
        fillcolor='rgba(29, 185, 84, 0.4)'
    ))
    
    # Add average listener comparison (simulated baseline)
    fig.add_trace(go.Scatterpolar(
        r=[0.55, 0.5, 0.6, 0.3, 0.55],
        theta=categories_closed,
        fill='toself',
        name='Avg Listener',
        line=dict(color='rgba(255, 255, 255, 0.3)', width=1, dash='dot'),
        fillcolor='rgba(255, 255, 255, 0.05)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                showticklabels=False,
                gridcolor='rgba(255, 255, 255, 0.1)'
            ),
            angularaxis=dict(
                tickfont=dict(size=14, color='white'),
                gridcolor='rgba(255, 255, 255, 0.1)'
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(color='white')
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=20, b=50, l=40, r=40),
        height=400
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h3 style='text-align: center; margin-bottom: 0;'>Your Audio DNA</h3>", unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Next →", use_container_width=True):
            st.session_state.slide = 5
            st.rerun()


def slide_genre(data):
    """Genre slide."""
    inject_css(GRADIENTS['genre'])
    
    top_genre = data['user_features'].get('top_genre', 'Pop')
    top_pct = data['user_features'].get('top_genre_percentage', 25)
    
    st.markdown(f"""
        <div class="center" style="padding-top: 10vh;">
            <p class="subtitle">You were all about</p>
            <h1 class="mega-title" style="text-transform: uppercase;">{top_genre}</h1>
            <p class="number-label">{top_pct:.0f}% of your music</p>
            <p class="fun-text">This genre was your happy place 🎸</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Top genres
    st.markdown("<br>", unsafe_allow_html=True)
    genres = list(data['genre_dist'].items())[:5]
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        for genre, pct in genres:
            st.markdown(f"""
                <div style="margin: 0.8rem 0;">
                    <div style="display: flex; justify-content: space-between;">
                        <span style="font-weight: 600; text-transform: capitalize;">{genre}</span>
                        <span>{pct:.1f}%</span>
                    </div>
                    <div class="progress-container">
                        <div class="progress-bar" style="width: {pct*2}%;"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Next →", use_container_width=True):
            st.session_state.slide = 6
            st.rerun()


def slide_personality(data):
    """Personality type slide."""
    inject_css(GRADIENTS['type'])
    
    user_type = data['user_type']
    uf = data['user_features']
    
    # Get the metrics that led to this classification
    energy = uf.get('avg_energy', 0.5)
    valence = uf.get('avg_valence', 0.5)
    danceability = uf.get('avg_danceability', 0.5)
    night_ratio = uf.get('night_listener_ratio', 0)
    genre_div = uf.get('genre_diversity', 0)
    acousticness = uf.get('avg_acousticness', 0.5)
    
    # Build the "why" explanation based on type
    type_name = user_type['type_name']
    why_text = ""
    
    if "Party Animal" in type_name:
        why_text = f"Your music is {int(danceability*100)}% danceable, {int(energy*100)}% energetic, and {int(valence*100)}% happy!"
    elif "Night Owl" in type_name:
        why_text = f"You add {int(night_ratio*100)}% of your music between 10PM-6AM 🌙"
    elif "Eclectic Explorer" in type_name:
        why_text = f"You explore {uf.get('total_genres', 0)} different genres - that's variety!"
    elif "Energy Seeker" in type_name:
        why_text = f"Your average energy is {int(energy*100)}% - you crave intensity!"
    else:  # Mood Rider
        why_text = f"Your happiness ranges widely - you ride the emotional waves"
    
    st.markdown(textwrap.dedent(f"""
        <div class="center" style="padding-top: 5vh;">
            <p class="subtitle">Based on your listening, you are</p>
        </div>
        
        <div class="personality-card">
            <p class="emoji-huge">{user_type['emoji']}</p>
            <h1 class="big-title" style="margin: 1rem 0;">{user_type['type_name'].upper()}</h1>
            <p class="subtitle" style="font-size: 1.2rem;">{user_type['description']}</p>
        <p style="color: rgba(255,255,255,0.7); margin-top: 1.5rem;">
        {' • '.join(user_type['traits'])}
        </p>
        </div>
        
        <p class="fun-text">{why_text}</p>
    """), unsafe_allow_html=True)
    
    # Show all personality types with explanations
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.expander("📋 What do the personality types mean?"):
        st.markdown("""
        **🎉 The Party Animal**  
        High danceability (>60%), high energy (>60%), high happiness (>50%)  
        *You love upbeat, danceable tracks that get people moving!*
        
        ---
        
        **🌙 The Night Owl**  
        Night listening ratio >30% (10PM - 6AM)  
        *You discover and add music when others are asleep*
        
        ---
        
        **🎸 The Eclectic Explorer**  
        High genre diversity (15+ genres)  
        *You don't stick to one style - you explore everything!*
        
        ---
        
        **🔥 The Energy Seeker**  
        High energy (>70%), low acousticness (<30%)  
        *You crave intense, powerful music regardless of mood*
        
        ---
        
        **🎭 The Mood Rider**  
        Default type - balanced or varied listening patterns  
        *Your music taste swings between different emotions*
        """)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Next →", use_container_width=True):
            st.session_state.slide = 7
            st.rerun()


def slide_compare(data):
    """Comparison/percentile slide."""
    inject_css(GRADIENTS['compare'])
    
    comparisons = data['comparison']['comparisons']
    insights = data['comparison']['insights']
    
    st.markdown("""
        <div class="center" style="padding-top: 5vh;">
            <p class="subtitle">How you compare</p>
            <h1 class="big-title">YOU'RE UNIQUE</h1>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        for comp in comparisons:
            pct = min(comp['percentile'], 100)
            st.markdown(f"""
                <div style="margin: 1rem 0;">
                    <div style="display: flex; justify-content: space-between;">
                        <span style="font-weight: 600;">{comp['metric']}</span>
                        <span style="font-weight: 700;">{comp['label']}</span>
                    </div>
                    <div class="progress-container">
                        <div class="progress-bar" style="width: {pct}%;"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        for insight in insights[:2]:
            st.markdown(f"""
                <div style="background: rgba(0,0,0,0.2); padding: 1rem; border-radius: 15px; margin: 0.5rem 0;">
                    <p style="margin: 0; font-size: 1.1rem;">{insight}</p>
                </div>
            """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("See Summary →", use_container_width=True):
            st.session_state.slide = 8
            st.rerun()


def slide_recommendations(data):
    """Recommendations/Future Hits slide."""
    inject_css(GRADIENTS['vibe'])
    
    st.markdown(textwrap.dedent(f"""
        <div class="center" style="padding-top: 5vh;">
        <p class="subtitle">Based on your {data['user_type']['type_name']} energy</p>
        <h1 class="mega-title">FUTURE HITS</h1>
        <p class="fun-text">Songs we think you'll love next 🔮</p>
        </div>
    """), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Mock recommendation logic: Get tracks from same dominant mood cluster
    mood_analyzer = data['mood_analyzer']
    df = data['df']
    
    # Get dominant mood cluster
    if mood_analyzer.fitted:
        clusters = mood_analyzer.fit(df) # Refit or use stored if possible, but fit is fast
        df['cluster'] = clusters
        
        # Simple logic: pick 5 random tracks from the dataset that are NOT in top 10 played (if we had play counts)
        # here just sample 5 random tracks form the dataframe to simulate 'discovery'
        recommendations = df.sample(5).to_dict('records')
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("<div class='glass' style='padding: 1rem;'>", unsafe_allow_html=True)
            
            for i, track in enumerate(recommendations):
                match_score = np.random.randint(85, 99) # Mock AI confidence
                track_name = track.get('track_name', 'Unknown Track')
                artist = track.get('artist_name(s)', 'Unknown Artist')
                
                st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 0.5rem;">
                        <div style="display: flex; align-items: center; gap: 1rem;">
                            <span style="font-size: 1.5rem; opacity: 0.5; font-weight: 700;">#{i+1}</span>
                            <div>
                                <p style="font-weight: 700; margin: 0; font-size: 1.1rem;">{track_name}</p>
                                <p style="margin: 0; font-size: 0.9rem; opacity: 0.8;">{artist}</p>
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <span style="background: #1DB954; color: black; padding: 2px 8px; border-radius: 10px; font-weight: 700; font-size: 0.8rem;">{match_score}% Match</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("See Summary →", use_container_width=True):
            st.session_state.slide = 9
            st.rerun()


def slide_share(data):
    """Final share slide."""
    inject_css(GRADIENTS['share'])
    
    stats = data['stats']
    user_type = data['user_type']
    top_artist = data['top_artists'].iloc[0]['artist_name(s)'] if len(data['top_artists']) > 0 else "Unknown"
    
    st.markdown(textwrap.dedent(f"""
        <div class="center" style="padding-top: 5vh;">
        <p style="font-size: 1.5rem; margin: 0;">Thanks for listening</p>
        <h1 class="mega-title">SPOTIFY WRAPPED</h1>
        </div>
        
        <div class="personality-card" style="background: rgba(0,0,0,0.25);">
            <p class="emoji-huge">{user_type['emoji']}</p>
            <h2 class="big-title" style="font-size: 2.5rem;">{user_type['type_name']}</h2>
            
        <div style="display: flex; justify-content: center; gap: 3rem; margin: 2rem 0;">
        <div>
        <p style="font-size: 2.5rem; font-weight: 900; margin: 0;">{stats['total_tracks']:,}</p>
        <p style="margin: 0; opacity: 0.8;">songs</p>
        </div>
        <div>
        <p style="font-size: 2.5rem; font-weight: 900; margin: 0;">{stats['total_hours']:.0f}</p>
        <p style="margin: 0; opacity: 0.8;">hours</p>
        </div>
        <div>
        <p style="font-size: 2.5rem; font-weight: 900; margin: 0;">{stats['total_artists']:,}</p>
        <p style="margin: 0; opacity: 0.8;">artists</p>
        </div>
        </div>
            
        <p style="font-size: 1.2rem; margin-top: 1rem;">Top Artist: <strong>{top_artist}</strong></p>
        </div>
        
        <p class="fun-text">See you next year 🎵</p>
    """), unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🔄 Start Over", use_container_width=True):
                st.session_state.slide = 0
                st.rerun()
        with col_b:
            if st.button("📂 New Data", use_container_width=True):
                st.session_state.uploaded_df = None
                st.session_state.slide = 0
                st.rerun()
    
    st.markdown("""
        <p style="text-align: center; color: rgba(255,255,255,0.6); margin-top: 2rem;">
            Spotify Wrapped Workshop • Aurora ISTE MANIPAL
        </p>
    """, unsafe_allow_html=True)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Initialize state
    if 'slide' not in st.session_state:
        st.session_state.slide = 0
    if 'uploaded_df' not in st.session_state:
        st.session_state.uploaded_df = None
    
    # Upload screen
    if st.session_state.uploaded_df is None:
        slide_upload()
        return
    
    # Load data
    data = load_wrapped_data_from_df(st.session_state.uploaded_df)
    if data is None:
        st.error("Failed to process data")
        if st.button("Try Again"):
            st.session_state.uploaded_df = None
            st.rerun()
        return
    
    # Slide navigation
    slide = st.session_state.slide
    
    if slide == 0:
        slide_upload()
    elif slide == 1:
        slide_intro(data)
    elif slide == 2:
        slide_stats(data)
    elif slide == 3:
        slide_top_artist(data)
    elif slide == 4:
        slide_vibe(data)
    elif slide == 5:
        slide_genre(data)
    elif slide == 6:
        slide_personality(data)
    elif slide == 7:
        slide_compare(data)
    elif slide == 8:
        slide_recommendations(data)
    elif slide == 9:
        slide_share(data)


if __name__ == "__main__":
    main()
