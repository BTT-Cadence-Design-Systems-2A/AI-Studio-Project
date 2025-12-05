# Streamlit Dashboard Setup Guide

## Overview
This Streamlit dashboard provides a user-friendly interface for analyzing Amazon product reviews and performing sentiment analysis.

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download NLTK data (if not already downloaded):**
   The dashboard will automatically download required NLTK data on first run, but you can also do it manually:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('wordnet')
   nltk.download('omw-1.4')
   ```

## Running the Dashboard

1. **Start the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

2. **Access the dashboard:**
   The dashboard will open in your default web browser at `http://localhost:8501`

## Features

### 🏠 Home
- Overview of the dashboard features
- Quick statistics

### 🔍 Sentiment Analysis
- **Model Selection**: Choose from multiple sentiment analysis models:
  - **Simple Rule-Based**: Fast keyword-based analysis (no ML required)
  - **BERT (Pretrained)**: State-of-the-art transformer model for accurate sentiment classification
  - **Twitter RoBERTa**: Optimized for social media and review text
- **Single Review**: Analyze individual reviews with sentiment classification and confidence scores
- **Batch Analysis**: Process multiple reviews at once with progress tracking
- View processed text and sentiment results
- Support for fine-tuned BERT models (advanced users)

### 📈 Data Exploration
- Load sample review data
- Filter by category and rating
- View detailed review information
- Explore dataset structure

### 📊 Statistics
- Comprehensive visualizations
- Sentiment distribution charts
- Rating analysis
- Category-wise statistics
- Text length analysis

### ⚙️ About
- Project information
- Team members
- Data source details

## Data Loading

The dashboard supports two modes:

1. **Sample Data**: Loads a small sample from the Amazon Reviews dataset (default)
2. **Full Dataset**: Loads data from HuggingFace (requires authentication)

For demonstration purposes, the dashboard includes mock data that will be used if the real data cannot be loaded.

## Notes

- **Transformer Models**: The dashboard now includes support for BERT and Twitter RoBERTa models
- **Model Loading**: Transformer models are loaded on-demand and cached for performance
- **Fine-tuned Models**: You can use your own fine-tuned BERT models by providing the model checkpoint path
- **Fallback**: If transformer libraries are not available, the dashboard automatically falls back to rule-based analysis
- Full dataset loading requires HuggingFace authentication

## Using Fine-tuned Models

If you have a fine-tuned BERT model from training (e.g., from the `Bert-base-cased-model` branch):

1. Save your model checkpoint to a local directory
2. In the dashboard, select "BERT (Pretrained)" as your model
3. Expand "Advanced: Use Fine-tuned Model"
4. Enter the path to your model checkpoint directory
5. The dashboard will use your fine-tuned model for predictions

Example model path: `/path/to/checkpoint-89250` or `./models/bert-finetuned`

## Troubleshooting

- **NLTK download errors**: Ensure you have internet connection for first-time setup
- **Data loading issues**: The dashboard will fall back to mock data if real data cannot be loaded
- **Port already in use**: Change the port with `streamlit run app.py --server.port 8502`

