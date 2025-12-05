"""
Utility functions for the Streamlit dashboard
"""
import pandas as pd
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import json
import fsspec
from itertools import islice
from collections import Counter
import re
import os
import torch
import numpy as np

# Transformer model imports (optional - will fail gracefully if not installed)
try:
    from transformers import (
        BertForSequenceClassification, 
        BertTokenizer,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        pipeline
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers library not available. Install with: pip install transformers torch")

# Download NLTK data (only once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4', quiet=True)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Configuration - matching notebook exactly
REPO = "McAuley-Lab/Amazon-Reviews-2023"
CATEGORIES = ["Software", "Video_Games", "All_Beauty"]
ALL_CATEGORIES = ["All_Beauty", "Amazon_Fashion", "Appliances", "Arts_Crafts_and_Sewing", "Automotive", "Baby_Products", "Beauty_and_Personal_Care", "Books",
              "CDs_and_Vinyl", "Cell_Phones_and_Accessories", "Clothing_Shoes_and_Jewelry", "Digital_Music", "Electronics", "Gift_Cards", "Grocery_and_Gourmet_Food",
              "Handmade_Products", "Health_and_Household", "Health_and_Personal_Care", "Home_and_Kitchen", "Industrial_and_Scientific",
              "Kindle_Store", "Magazine_Subscriptions", "Movies_and_TV", "Musical_Instruments", "Office_Products", "Patio_Lawn_and_Garden", "Pet_Supplies",
              "Software", "Sports_and_Outdoors", "Subscription_Boxes", "Tools_and_Home_Improvement", "Toys_and_Games", "Video_Games",
              "Unknown"]
N_PER_CAT = 1000  # Reduced for faster loading in dashboard
N_META = 60_000


def remove_punctuation(text: str) -> str:
    """
    Remove all punctuation from a string
    """
    if not isinstance(text, str):
        return ""
    return text.translate(str.maketrans("", "", string.punctuation))


def preprocess_text(text: str) -> str:
    """
    Preprocess text: lowercase, remove punctuation, normalize whitespace
    Matches the exact pipeline from the notebook (Cell 21)
    """
    if not isinstance(text, str):
        return ""
    
    # Exact pipeline from notebook: .str.lower().apply(remove_punctuation).str.replace(r"\s+", " ", regex=True).str.strip()
    text = text.lower()
    text = remove_punctuation(text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    
    return text


def create_clean_review(text: str) -> str:
    """
    Creates clean_review exactly as in notebook (Cell 21)
    """
    if not isinstance(text, str):
        return ""
    return preprocess_text(text)


def create_clean_title(title: str) -> str:
    """
    Creates clean_title exactly as in notebook (Cell 21)
    """
    if not isinstance(title, str):
        return ""
    return preprocess_text(title)


def lemmatize_text(text: str) -> str:
    """
    Lemmatize text using WordNetLemmatizer
    Exact implementation from notebook (Cell 23)
    """
    if not isinstance(text, str):
        return ""
    
    try:
        tokens = word_tokenize(text)
        lemmas = [lemmatizer.lemmatize(token) for token in tokens]
        return " ".join(lemmas)
    except Exception as e:
        return text  # Return original text if lemmatization fails


def tokenize_review(text: str) -> list:
    """
    Tokenize review text using word_tokenize
    Matches notebook Cell 33
    """
    if not isinstance(text, str):
        return []
    try:
        return word_tokenize(text)
    except Exception:
        return []


def create_sentiment_label(rating: float) -> str:
    """
    Create sentiment label from rating
    Exact implementation from notebook (Cell 26)
    - 4-5 stars: positive
    - 3 stars: neutral
    - 1-2 stars: negative
    """
    if rating >= 4:
        return 'positive'
    elif rating <= 2:
        return 'negative'
    else:
        return 'neutral'


def analyze_sentiment_simple(text: str) -> tuple[str, str]:
    """
    Simple rule-based sentiment analysis
    Returns (sentiment, confidence)
    """
    if not text:
        return 'neutral', 'low'
    
    text_lower = text.lower()
    
    # Positive keywords
    positive_words = [
        'excellent', 'great', 'good', 'amazing', 'wonderful', 'love', 'perfect',
        'fantastic', 'awesome', 'outstanding', 'brilliant', 'superb', 'best',
        'satisfied', 'happy', 'pleased', 'recommend', 'highly', 'quality'
    ]
    
    # Negative keywords
    negative_words = [
        'bad', 'terrible', 'awful', 'horrible', 'worst', 'disappointed',
        'poor', 'waste', 'broken', 'defective', 'faulty', 'useless',
        'hate', 'regret', 'avoid', 'problem', 'issue', 'complaint'
    ]
    
    # Count occurrences
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    # Determine sentiment
    if positive_count > negative_count:
        return 'positive', 'medium'
    elif negative_count > positive_count:
        return 'negative', 'medium'
    else:
        return 'neutral', 'low'


# Global model cache to avoid reloading models
_model_cache = {}


def load_bert_model(model_path: str = None, model_name: str = "bert-base-cased"):
    """
    Load BERT model for sentiment analysis
    If model_path is provided, loads fine-tuned model, otherwise loads pretrained
    
    Args:
        model_path: Path to fine-tuned model checkpoint (optional)
        model_name: HuggingFace model name (default: bert-base-cased)
    
    Returns:
        (model, tokenizer) tuple or None if loading fails
    """
    if not TRANSFORMERS_AVAILABLE:
        return None, None
    
    cache_key = f"bert_{model_path or model_name}"
    
    if cache_key in _model_cache:
        return _model_cache[cache_key]
    
    try:
        if model_path and model_path.strip() and os.path.exists(model_path.strip()):
            # Load fine-tuned model
            model_path = model_path.strip()
            model = BertForSequenceClassification.from_pretrained(model_path)
            tokenizer = BertTokenizer.from_pretrained(model_path)
        else:
            # Load pretrained model
            model = BertForSequenceClassification.from_pretrained(
                model_name, 
                num_labels=3
            )
            tokenizer = BertTokenizer.from_pretrained(model_name)
        
        model.eval()  # Set to evaluation mode
        _model_cache[cache_key] = (model, tokenizer)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading BERT model: {e}")
        return None, None


def load_twitter_roberta_model(model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
    """
    Load Twitter RoBERTa model for sentiment analysis
    
    Args:
        model_name: HuggingFace model name
    
    Returns:
        (model, tokenizer) tuple or None if loading fails
    """
    if not TRANSFORMERS_AVAILABLE:
        return None, None
    
    cache_key = f"roberta_{model_name}"
    
    if cache_key in _model_cache:
        return _model_cache[cache_key]
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()
        _model_cache[cache_key] = (model, tokenizer)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading Twitter RoBERTa model: {e}")
        return None, None


def load_sentiment_pipeline(model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
    """
    Load HuggingFace pipeline for sentiment analysis (easiest to use)
    
    Args:
        model_name: HuggingFace model name
    
    Returns:
        Pipeline object or None if loading fails
    """
    if not TRANSFORMERS_AVAILABLE:
        return None
    
    cache_key = f"pipeline_{model_name}"
    
    if cache_key in _model_cache:
        return _model_cache[cache_key]
    
    try:
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1
        )
        _model_cache[cache_key] = sentiment_pipeline
        return sentiment_pipeline
    except Exception as e:
        print(f"Error loading sentiment pipeline: {e}")
        return None


def analyze_sentiment_bert(text: str, model_path: str = None, model_name: str = "bert-base-cased") -> tuple[str, float]:
    """
    Analyze sentiment using BERT model
    Returns (sentiment, confidence_score)
    
    Args:
        text: Review text to analyze
        model_path: Path to fine-tuned model (optional)
        model_name: HuggingFace model name
    
    Returns:
        (sentiment_label, confidence) tuple
    """
    if not TRANSFORMERS_AVAILABLE:
        return analyze_sentiment_simple(text)[0], 0.5
    
    model, tokenizer = load_bert_model(model_path, model_name)
    
    if model is None or tokenizer is None:
        return analyze_sentiment_simple(text)[0], 0.5
    
    try:
        # Preprocess text (use clean_review pipeline)
        clean_text = create_clean_review(text)
        
        # Tokenize
        inputs = tokenizer(
            clean_text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class = logits.argmax(dim=-1).item()
            confidence = probs[0][predicted_class].item()
        
        # Map to sentiment labels (0: negative, 1: neutral, 2: positive)
        label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        sentiment = label_map.get(predicted_class, 'neutral')
        
        return sentiment, confidence
    except Exception as e:
        print(f"Error in BERT prediction: {e}")
        return analyze_sentiment_simple(text)[0], 0.5


def analyze_sentiment_roberta(text: str, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest") -> tuple[str, float]:
    """
    Analyze sentiment using Twitter RoBERTa model
    Returns (sentiment, confidence_score)
    
    Args:
        text: Review text to analyze
        model_name: HuggingFace model name
    
    Returns:
        (sentiment_label, confidence) tuple
    """
    if not TRANSFORMERS_AVAILABLE:
        return analyze_sentiment_simple(text)[0], 0.5
    
    # Try using pipeline first (easier)
    pipeline_model = load_sentiment_pipeline(model_name)
    
    if pipeline_model:
        try:
            result = pipeline_model(text)[0]
            label = result['label'].lower()
            score = result['score']
            
            # Map labels to our format
            if 'positive' in label or 'pos' in label:
                return 'positive', score
            elif 'negative' in label or 'neg' in label:
                return 'negative', score
            else:
                return 'neutral', score
        except Exception as e:
            print(f"Error in RoBERTa pipeline prediction: {e}")
    
    # Fallback to direct model loading
    model, tokenizer = load_twitter_roberta_model(model_name)
    
    if model is None or tokenizer is None:
        return analyze_sentiment_simple(text)[0], 0.5
    
    try:
        inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class = logits.argmax(dim=-1).item()
            confidence = probs[0][predicted_class].item()
        
        # Twitter RoBERTa typically has: LABEL_0 (negative), LABEL_1 (neutral), LABEL_2 (positive)
        # But this can vary, so we'll use the class with highest probability
        label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        sentiment = label_map.get(predicted_class, 'neutral')
        
        return sentiment, confidence
    except Exception as e:
        print(f"Error in RoBERTa prediction: {e}")
        return analyze_sentiment_simple(text)[0], 0.5


def analyze_sentiment(text: str, model_type: str = "simple", model_path: str = None) -> tuple[str, float]:
    """
    Unified sentiment analysis function that supports multiple models
    
    Args:
        text: Review text to analyze
        model_type: One of "simple", "bert", "roberta", "bert-finetuned"
        model_path: Path to fine-tuned model (required for bert-finetuned)
    
    Returns:
        (sentiment_label, confidence_score) tuple
    """
    if not text:
        return 'neutral', 0.0
    
    if model_type == "simple":
        sentiment, conf_str = analyze_sentiment_simple(text)
        # Convert confidence string to float
        conf_map = {'low': 0.3, 'medium': 0.6, 'high': 0.9}
        confidence = conf_map.get(conf_str, 0.5)
        return sentiment, confidence
    
    elif model_type == "bert" or model_type == "bert-pretrained":
        return analyze_sentiment_bert(text, model_name="bert-base-cased")
    
    elif model_type == "bert-finetuned":
        if not model_path:
            st.warning("Model path required for fine-tuned BERT. Using pretrained BERT instead.")
            return analyze_sentiment_bert(text, model_name="bert-base-cased")
        return analyze_sentiment_bert(text, model_path=model_path)
    
    elif model_type == "roberta" or model_type == "twitter-roberta":
        return analyze_sentiment_roberta(text)
    
    else:
        # Default to simple
        return analyze_sentiment_simple(text)[0], 0.5


def stream_jsonl(url: str, limit: int | None = None):
    """
    Stream a JSONL file line-by-line from Hugging Face
    """
    try:
        with fsspec.open(url, "rt") as f:
            for idx, line in enumerate(f):
                if limit is not None and idx >= limit:
                    break
                obj = json.loads(line)
                
                if "price" in obj and obj["price"] is not None:
                    obj["price"] = str(obj["price"])
                
                yield obj
    except Exception as e:
        print(f"Error streaming JSONL: {e}")
        return


def load_category_into_review(category: str, n_reviews: int):
    """
    Load one category's reviews as DataFrame
    Exact implementation from notebook (Cell 14)
    """
    try:
        reviews_url = f"hf://datasets/{REPO}/raw/review_categories/{category}.jsonl"
        
        data = (
            {k: row.get(k) for k in ["rating", "title", "text"]}
            for row in islice(stream_jsonl(reviews_url), n_reviews)
        )
        
        reviews_df = pd.DataFrame(data).assign(category=category)
        return reviews_df
    except Exception as e:
        print(f"Error loading category {category}: {e}")
        return pd.DataFrame()


def load_sample_data(n_reviews_per_cat: int = 100):
    """
    Load sample review data for the dashboard
    Matches notebook approach (Cell 15) but with smaller sample size
    """
    try:
        all_reviews = []
        
        # Use CATEGORIES for faster loading (can switch to ALL_CATEGORIES for full dataset)
        for cat in CATEGORIES:
            try:
                r_df = load_category_into_review(cat, n_reviews=n_reviews_per_cat)
                if not r_df.empty:
                    all_reviews.append(r_df)
            except Exception as e:
                print(f"Could not load category {cat}: {e}")
                continue
        
        if all_reviews:
            reviews_df = pd.concat(all_reviews, ignore_index=True)
            
            # Apply the exact preprocessing pipeline from notebook
            # Cell 21: clean_review and clean_title
            reviews_df['clean_review'] = reviews_df['text'].apply(create_clean_review)
            reviews_df['clean_title'] = reviews_df['title'].apply(create_clean_title)
            
            # Cell 24: lemmatized_review and lemmatized_title
            reviews_df['lemmatized_review'] = reviews_df['clean_review'].apply(lemmatize_text)
            reviews_df['lemmatized_title'] = reviews_df['clean_title'].apply(lemmatize_text)
            
            # Cell 27: sentiment_labels
            reviews_df['sentiment_labels'] = reviews_df['rating'].apply(create_sentiment_label)
            
            # Cell 33: tokenized_review
            reviews_df['tokenized_review'] = reviews_df['clean_review'].apply(tokenize_review)
            
            # Ensure required columns exist
            required_cols = ['rating', 'title', 'text', 'category']
            for col in required_cols:
                if col not in reviews_df.columns:
                    reviews_df[col] = None
            
            # Remove rows with missing essential data
            reviews_df = reviews_df.dropna(subset=['text', 'rating'])
            
            return reviews_df
        else:
            # Return mock data if loading fails
            return create_mock_data()
    
    except Exception as e:
        print(f"Error loading sample data: {e}")
        return create_mock_data()


def create_mock_data():
    """
    Create mock review data for demonstration when real data cannot be loaded
    """
    mock_reviews = [
        {
            'rating': 5.0,
            'title': 'Excellent product!',
            'text': 'This product exceeded my expectations. The quality is outstanding and I would highly recommend it to anyone.',
            'category': 'Software',
            'asin': 'MOCK001'
        },
        {
            'rating': 4.0,
            'title': 'Great value for money',
            'text': 'Good product overall. Works as expected and the price is reasonable. Some minor improvements could be made.',
            'category': 'Software',
            'asin': 'MOCK002'
        },
        {
            'rating': 3.0,
            'title': 'Average product',
            'text': 'The product is okay. Nothing special, but it does the job. Could be better.',
            'category': 'Video_Games',
            'asin': 'MOCK003'
        },
        {
            'rating': 2.0,
            'title': 'Disappointed',
            'text': 'Not what I expected. The product has several issues and the quality is poor. Would not recommend.',
            'category': 'Video_Games',
            'asin': 'MOCK004'
        },
        {
            'rating': 1.0,
            'title': 'Terrible experience',
            'text': 'This is the worst product I have ever purchased. It broke immediately and customer service was unhelpful.',
            'category': 'All_Beauty',
            'asin': 'MOCK005'
        },
        {
            'rating': 5.0,
            'title': 'Amazing quality!',
            'text': 'I love this product! It works perfectly and the design is beautiful. Worth every penny.',
            'category': 'All_Beauty',
            'asin': 'MOCK006'
        },
        {
            'rating': 4.0,
            'title': 'Very satisfied',
            'text': 'Great product with good features. Minor issues but overall very happy with the purchase.',
            'category': 'Software',
            'asin': 'MOCK007'
        },
        {
            'rating': 2.0,
            'title': 'Not worth it',
            'text': 'Poor quality and does not work as advertised. Regret buying this product.',
            'category': 'Video_Games',
            'asin': 'MOCK008'
        }
    ]
    
    return pd.DataFrame(mock_reviews)

