"""
EDA ANALYSIS - Comprehensive Exploratory Data Analysis
Visualization and insights from Spotify listening data
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

from data_pipeline import SpotifyDataPipeline


class SpotifyEDA:
    """
    Comprehensive EDA for Spotify listening data.
    
    Visualizations:
    - Listening patterns over time
    - Audio feature distributions
    - Genre analysis
    - Artist frequency
    - Mood clustering preview
    """
    
    # Spotify-inspired color palette
    COLORS = {
        'primary': '#1DB954',      # Spotify green
        'secondary': '#191414',     # Spotify black
        'accent1': '#FF6B6B',       # Coral
        'accent2': '#4ECDC4',       # Teal
        'accent3': '#FFE66D',       # Yellow
        'accent4': '#95E1D3',       # Mint
        'gradient': ['#1DB954', '#1ed760', '#4ECDC4', '#FFE66D', '#FF6B6B']
    }
    
    def __init__(self, csv_path: str = "data/SytheticData1000.csv"):
        """
        Initialize EDA with data pipeline.
        
        Args:
            csv_path: Path to Liked_Songs.csv
        """
        self.pipeline = SpotifyDataPipeline(csv_path)
        self.df = self.pipeline.process_data()
        
    def plot_listening_timeline(self) -> go.Figure:
        """
        Create listening timeline chart (tracks added over time).
        
        Returns:
            Plotly figure
        """
        # Group by date
        daily = self.df.groupby('added_date').size().reset_index(name='tracks')
        daily['added_date'] = pd.to_datetime(daily['added_date'])
        
        # 7-day rolling average
        daily['rolling_avg'] = daily['tracks'].rolling(7, min_periods=1).mean()
        
        fig = go.Figure()
        
        # Bar chart for daily tracks
        fig.add_trace(go.Bar(
            x=daily['added_date'],
            y=daily['tracks'],
            name='Daily Tracks',
            marker_color='rgba(29, 185, 84, 0.4)',
            hovertemplate='%{x|%b %d, %Y}<br>%{y} tracks<extra></extra>'
        ))
        
        # Line for rolling average
        fig.add_trace(go.Scatter(
            x=daily['added_date'],
            y=daily['rolling_avg'],
            mode='lines',
            name='7-Day Average',
            line=dict(color='#1DB954', width=3),
            hovertemplate='%{x|%b %d, %Y}<br>Avg: %{y:.1f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='🎵 Your Listening Timeline',
            xaxis_title='Date',
            yaxis_title='Tracks Added',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.1)',
            hovermode='x unified',
            height=400
        )
        
        return fig
    
    def plot_hour_heatmap(self) -> go.Figure:
        """
        Create hour vs day heatmap of listening activity.
        
        Returns:
            Plotly figure
        """
        # Create pivot table
        heatmap_data = self.df.groupby(['added_day', 'added_hour']).size().unstack(fill_value=0)
        
        # Day labels
        day_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=list(range(24)),
            y=day_labels,
            colorscale=[[0, 'rgb(25, 20, 20)'], [0.5, 'rgb(29, 185, 84)'], [1, 'rgb(30, 215, 96)']],
            hovertemplate='%{y} at %{x}:00<br>%{z} tracks<extra></extra>'
        ))
        
        fig.update_layout(
            title='⏰ When You Add Music',
            xaxis_title='Hour of Day',
            yaxis_title='Day of Week',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            height=350
        )
        
        return fig
    
    def plot_audio_features_radar(self) -> go.Figure:
        """
        Create radar chart of average audio features.
        
        Returns:
            Plotly figure
        """
        features = ['energy', 'valence', 'danceability', 'acousticness', 'speechiness', 'liveness']
        available_features = [f for f in features if f in self.df.columns]
        
        values = [self.df[f].mean() for f in available_features]
        values.append(values[0])  # Close the polygon
        
        fig = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=available_features + [available_features[0]],
            fill='toself',
            name='Your Profile',
            line_color='#1DB954',
            fillcolor='rgba(29, 185, 84, 0.3)',
            marker=dict(size=8, color='#1DB954')
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], gridcolor='rgba(255,255,255,0.2)'),
                angularaxis=dict(gridcolor='rgba(255,255,255,0.2)')
            ),
            title='🎭 Your Audio DNA',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            height=450
        )
        
        return fig
    
    def plot_audio_features_distribution(self) -> go.Figure:
        """
        Create histograms for key audio features.
        
        Returns:
            Plotly figure
        """
        features = ['energy', 'valence', 'danceability', 'tempo']
        available_features = [f for f in features if f in self.df.columns]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[f.capitalize() for f in available_features]
        )
        
        colors = ['#1DB954', '#4ECDC4', '#FFE66D', '#FF6B6B']
        
        for i, feature in enumerate(available_features):
            row = i // 2 + 1
            col = i % 2 + 1
            
            fig.add_trace(
                go.Histogram(
                    x=self.df[feature],
                    name=feature.capitalize(),
                    marker_color=colors[i],
                    opacity=0.8,
                    nbinsx=30
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title='📊 Audio Feature Distributions',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            height=500
        )
        
        return fig
    
    def plot_genre_sunburst(self) -> go.Figure:
        """
        Create sunburst chart of genre distribution.
        
        Returns:
            Plotly figure
        """
        genre_dist = self.pipeline.get_genre_distribution()
        
        # Take top 15 genres
        top_genres = dict(list(genre_dist.items())[:15])
        other_pct = sum(list(genre_dist.values())[15:]) if len(genre_dist) > 15 else 0
        if other_pct > 0:
            top_genres['Other'] = other_pct
        
        fig = go.Figure(go.Sunburst(
            labels=list(top_genres.keys()),
            values=list(top_genres.values()),
            marker=dict(
                colors=self.COLORS['gradient'] * 5,
                line=dict(color='white', width=2)
            ),
            textfont=dict(size=14, color='white'),
            hovertemplate='<b>%{label}</b><br>%{value:.1f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title='🎸 Your Genre Universe',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            height=500,
            margin=dict(t=80, l=0, r=0, b=0)
        )
        
        return fig
    
    def plot_top_artists(self, n: int = 10) -> go.Figure:
        """
        Create horizontal bar chart of top artists.
        
        Args:
            n: Number of top artists to show
            
        Returns:
            Plotly figure
        """
        top_artists = self.pipeline.get_top_artists(n)
        
        # Reverse for horizontal bar chart (top at top)
        top_artists = top_artists.iloc[::-1]
        
        fig = go.Figure(go.Bar(
            x=top_artists['track_count'],
            y=top_artists['artist_name(s)'],
            orientation='h',
            marker=dict(
                color=top_artists['track_count'],
                colorscale=[[0, '#4ECDC4'], [0.5, '#1DB954'], [1, '#1ed760']],
                line=dict(color='white', width=1)
            ),
            text=top_artists['track_count'],
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>%{x} tracks<extra></extra>'
        ))
        
        fig.update_layout(
            title='🎤 Your Top Artists',
            xaxis_title='Track Count',
            yaxis_title='',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            height=450
        )
        
        return fig
    
    def plot_mood_quadrant(self) -> go.Figure:
        """
        Create energy vs valence scatter plot (mood quadrant).
        
        Returns:
            Plotly figure
        """
        # Sample data if too large
        df_sample = self.df.sample(min(500, len(self.df)), random_state=42)
        
        # Assign mood labels based on energy and valence
        def get_mood(row):
            if row['energy'] > 0.5 and row['valence'] > 0.5:
                return 'Happy & Energetic'
            elif row['energy'] > 0.5 and row['valence'] <= 0.5:
                return 'Angry & Intense'
            elif row['energy'] <= 0.5 and row['valence'] > 0.5:
                return 'Peaceful & Happy'
            else:
                return 'Sad & Calm'
        
        df_sample['mood'] = df_sample.apply(get_mood, axis=1)
        
        color_map = {
            'Happy & Energetic': '#1DB954',
            'Angry & Intense': '#FF6B6B',
            'Peaceful & Happy': '#FFE66D',
            'Sad & Calm': '#4ECDC4'
        }
        
        fig = px.scatter(
            df_sample,
            x='energy',
            y='valence',
            color='mood',
            color_discrete_map=color_map,
            hover_data=['track_name', 'artist_name(s)'],
            opacity=0.7
        )
        
        # Add quadrant lines
        fig.add_hline(y=0.5, line_dash='dash', line_color='rgba(255,255,255,0.3)')
        fig.add_vline(x=0.5, line_dash='dash', line_color='rgba(255,255,255,0.3)')
        
        # Add quadrant labels
        annotations = [
            dict(x=0.75, y=0.85, text='Happy & Energetic', showarrow=False, font=dict(size=12, color='white')),
            dict(x=0.75, y=0.15, text='Angry & Intense', showarrow=False, font=dict(size=12, color='white')),
            dict(x=0.25, y=0.85, text='Peaceful & Happy', showarrow=False, font=dict(size=12, color='white')),
            dict(x=0.25, y=0.15, text='Sad & Calm', showarrow=False, font=dict(size=12, color='white')),
        ]
        
        fig.update_layout(
            title='🎭 Your Music Mood Map',
            xaxis_title='Energy →',
            yaxis_title='Happiness (Valence) →',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            height=500,
            annotations=annotations
        )
        
        return fig
    
    def plot_tempo_distribution(self) -> go.Figure:
        """
        Create tempo distribution histogram with genre color coding.
        
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=self.df['tempo'],
            nbinsx=50,
            marker_color='#1DB954',
            opacity=0.8,
            hovertemplate='%{x:.0f} BPM<br>%{y} tracks<extra></extra>'
        ))
        
        # Add average line
        avg_tempo = self.df['tempo'].mean()
        fig.add_vline(
            x=avg_tempo,
            line_dash='dash',
            line_color='#FFE66D',
            annotation_text=f'Avg: {avg_tempo:.0f} BPM',
            annotation_position='top right'
        )
        
        fig.update_layout(
            title='🥁 Tempo Distribution',
            xaxis_title='Tempo (BPM)',
            yaxis_title='Track Count',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            height=350
        )
        
        return fig
    
    def plot_feature_correlation(self) -> go.Figure:
        """
        Create correlation heatmap of audio features.
        
        Returns:
            Plotly figure
        """
        features = ['energy', 'valence', 'danceability', 'acousticness', 
                   'speechiness', 'instrumentalness', 'liveness', 'tempo', 'loudness']
        available_features = [f for f in features if f in self.df.columns]
        
        corr_matrix = self.df[available_features].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=available_features,
            y=available_features,
            colorscale=[[0, '#FF6B6B'], [0.5, '#191414'], [1, '#1DB954']],
            zmid=0,
            hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='🔗 Feature Correlations',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            height=450
        )
        
        return fig
    
    def plot_monthly_trends(self) -> go.Figure:
        """
        Create monthly listening trends chart.
        
        Returns:
            Plotly figure
        """
        monthly = self.pipeline.get_monthly_taste_vectors()
        monthly['year_month'] = monthly['year_month'].astype(str)
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Track Count per Month', 'Energy & Valence Trends'],
            vertical_spacing=0.15
        )
        
        # Track count
        fig.add_trace(
            go.Bar(
                x=monthly['year_month'],
                y=monthly['track_count'],
                name='Tracks',
                marker_color='#1DB954'
            ),
            row=1, col=1
        )
        
        # Energy and valence trends
        if 'energy' in monthly.columns:
            fig.add_trace(
                go.Scatter(
                    x=monthly['year_month'],
                    y=monthly['energy'],
                    mode='lines+markers',
                    name='Energy',
                    line=dict(color='#FF6B6B', width=2)
                ),
                row=2, col=1
            )
        
        if 'valence' in monthly.columns:
            fig.add_trace(
                go.Scatter(
                    x=monthly['year_month'],
                    y=monthly['valence'],
                    mode='lines+markers',
                    name='Valence',
                    line=dict(color='#4ECDC4', width=2)
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            title='📅 Monthly Listening Trends',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            height=600,
            showlegend=True
        )
        
        return fig
    
    def generate_stats_summary(self) -> Dict:
        """
        Generate comprehensive statistics summary.
        
        Returns:
            Dictionary of stats
        """
        stats = self.pipeline.get_listening_stats()
        user_features = self.pipeline.calculate_user_features()
        
        return {
            **stats,
            **user_features
        }
    
    def run_full_eda(self, save_path: Optional[str] = None):
        """
        Run full EDA and display all visualizations.
        
        Args:
            save_path: Optional path to save HTML report
        """
        print("=" * 60)
        print("🎵 SPOTIFY WRAPPED - EXPLORATORY DATA ANALYSIS")
        print("=" * 60)
        
        # Stats summary
        stats = self.generate_stats_summary()
        print(f"\n📊 LISTENING STATS")
        print(f"   Total Tracks: {stats['total_tracks']:,}")
        print(f"   Total Artists: {stats['total_artists']:,}")
        print(f"   Total Hours: {stats['total_hours']:.1f}")
        print(f"   Date Range: {stats['date_range_start']} to {stats['date_range_end']}")
        
        print(f"\n🎭 AUDIO PROFILE")
        for feature in ['avg_energy', 'avg_valence', 'avg_danceability', 'avg_tempo']:
            if feature in stats:
                print(f"   {feature.replace('avg_', '').capitalize()}: {stats[feature]:.2f}")
        
        print(f"\n🎸 TOP GENRE: {stats['top_genre']} ({stats['top_genre_percentage']:.1f}%)")
        print(f"   Genre Diversity: {stats['genre_diversity']:.3f}")
        
        print(f"\n⏰ LISTENING PATTERNS")
        print(f"   Night Listener: {stats['night_listener_ratio']*100:.1f}%")
        print(f"   Weekend Listener: {stats['weekend_ratio']*100:.1f}%")
        
        # Generate all plots
        figs = {
            'timeline': self.plot_listening_timeline(),
            'heatmap': self.plot_hour_heatmap(),
            'radar': self.plot_audio_features_radar(),
            'distributions': self.plot_audio_features_distribution(),
            'genres': self.plot_genre_sunburst(),
            'artists': self.plot_top_artists(),
            'mood': self.plot_mood_quadrant(),
            'tempo': self.plot_tempo_distribution(),
            'correlation': self.plot_feature_correlation(),
            'monthly': self.plot_monthly_trends()
        }
        
        print(f"\n✅ Generated {len(figs)} visualizations")
        
        if save_path:
            # Save as HTML report
            from plotly.subplots import make_subplots
            import plotly.io as pio
            
            # Combine into single HTML
            html_content = "<html><head><title>Spotify EDA Report</title></head><body style='background:#191414;color:white;'>"
            for name, fig in figs.items():
                html_content += f"<h2>{name.upper()}</h2>"
                html_content += pio.to_html(fig, include_plotlyjs='cdn', full_html=False)
            html_content += "</body></html>"
            
            with open(save_path, 'w') as f:
                f.write(html_content)
            print(f"📁 Saved report to {save_path}")
        
        return figs, stats


# Usage Example
if __name__ == "__main__":
    # Run EDA
    eda = SpotifyEDA("data/SytheticData1000.csv")
    figs, stats = eda.run_full_eda()
    
    # Show first figure
    print("\n📈 Showing listening timeline...")
    figs['timeline'].show()
