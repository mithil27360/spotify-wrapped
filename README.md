# 🎵 Spotify Wrapped AI

A real Spotify Wrapped-style story experience with AI/ML analysis of your listening data.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Features

- **Story-Flow Experience**: 9 interactive slides mimicking Spotify Wrapped
- **Big Numbers**: Your stats displayed with stunning visuals
- **Mood Analysis**: AI-powered vibe detection with radar charts
- **User Personality**: Are you an Explorer, Loyalist, or Party Animal?
- **Percentile Rankings**: See how you compare to other listeners
- **Personalized Recommendations**: ML-powered track suggestions

## Project Structure

```
spotify-wrapped/
├── app.py                 # Main Streamlit story-flow app
├── requirements.txt       # Python dependencies
├── .gitignore
├── data/
│   └── SytheticData1000.csv    # Your Spotify listening data
├── src/
│   ├── data_pipeline.py   # Data loading & feature engineering
│   └── eda_analysis.py    # EDA visualizations
├── ml/
│   ├── user_classifier.py    # User personality classification
│   ├── mood_analyzer.py      # K-Means mood clustering
│   ├── genre_classifier.py   # Random Forest genre prediction
│   ├── taste_matcher.py      # ALS matrix factorization
│   ├── hybrid_recommender.py # Hybrid recommendation engine
│   └── model_trainer.py      # Model training pipeline
├── api/
│   └── fastapi_backend.py    # FastAPI REST API
└── docs/
    ├── IMPLEMENTATION_SUMMARY.md
    ├── PROJECT_STRUCTURE.md
    ├── QUICK_REFERENCE.md
    └── STREAMLIT_GUIDE.md
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the App

```bash
streamlit run app.py
```

### 3. Open in Browser

Navigate to [http://localhost:8501](http://localhost:8501)

## ML Models

| Model | Purpose | Algorithm |
|-------|---------|-----------|
| **TasteMatcher** | User-track taste matching | ALS Matrix Factorization |
| **MoodAnalyzer** | Mood/vibe detection | K-Means Clustering |
| **GenreClassifier** | Genre prediction | Random Forest |
| **UserClassifier** | Personality types | Rule-based + K-Means |
| **HybridRecommender** | Track recommendations | CF + Content-Based |

## Your Stats

Based on your `SytheticData1000.csv`:

- **2,414** tracks analyzed
- **153** hours of music
- **1,725** unique artists
- **Personality**: The Party Animal 🎉

## Tech Stack

- **Frontend**: Streamlit + Plotly
- **Backend**: FastAPI
- **ML**: scikit-learn, implicit
- **Data**: pandas, numpy

## License

MIT License
