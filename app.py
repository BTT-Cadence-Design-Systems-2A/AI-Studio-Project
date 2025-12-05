import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import (
    preprocess_text,
    create_sentiment_label,
    load_sample_data,
    analyze_sentiment_simple,
    analyze_sentiment,
    lemmatize_text,
    tokenize_review,
    create_clean_review,
    create_clean_title,
    TRANSFORMERS_AVAILABLE
)
import json
import fsspec
from itertools import islice

# Page configuration
st.set_page_config(
    page_title="Amazon Review Sentiment Analysis Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Amazon Review Sentiment Analysis Dashboard - Powered by AI"
    }
)

# Ensure something always renders
try:
    st.write("✅ Dashboard is running...")
except Exception as e:
    st.error(f"Error: {e}")

# Custom CSS with Amazon colors and modern design
st.markdown("""
    <style>
    /* Amazon Color Palette */
    :root {
        --amazon-orange: #FF9900;
        --amazon-dark: #232F3E;
        --amazon-blue: #146EB4;
        --amazon-light: #F3F3F3;
        --amazon-text: #111111;
    }
    
    /* Main App Styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header Styling */
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #FF9900 0%, #FFB84D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        padding: 1.5rem 0;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #232F3E 0%, #1a2532 100%);
    }
    
    /* Sidebar text - make all text white/visible */
    [data-testid="stSidebar"] {
        color: white !important;
    }
    
    [data-testid="stSidebar"] h1 {
        color: #FF9900 !important;
        font-weight: 700;
    }
    
    /* Sidebar radio button labels - ensure visibility */
    [data-testid="stSidebar"] .stRadio label {
        color: #FFFFFF !important;
        font-weight: 500;
    }
    
    [data-testid="stSidebar"] .stRadio label div {
        color: #FFFFFF !important;
    }
    
    /* Sidebar text elements */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] div:not([class*="st"]):not([data-baseweb]),
    [data-testid="stSidebar"] span {
        color: #FFFFFF !important;
    }
    
    /* Radio button text specifically */
    [data-testid="stSidebar"] [data-baseweb="radio"] label {
        color: #FFFFFF !important;
    }
    
    /* Selected radio button - orange highlight */
    [data-testid="stSidebar"] [data-baseweb="radio"] input:checked ~ label {
        color: #FF9900 !important;
        font-weight: 600;
    }
    
    /* Make sure all text in sidebar is visible */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stRadio > label,
    [data-testid="stSidebar"] .stRadio > div > label {
        color: #FFFFFF !important;
    }
    
    /* Button Styling - Amazon Orange with hover effects */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #FF9900 0%, #FFB84D 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 6px rgba(255, 153, 0, 0.3);
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }
    
    .stButton>button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.2);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stButton>button:hover::before {
        width: 300px;
        height: 300px;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #FFB84D 0%, #FF9900 100%);
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 16px rgba(255, 153, 0, 0.5);
    }
    
    .stButton>button:active {
        transform: translateY(-1px) scale(0.98);
        box-shadow: 0 4px 8px rgba(255, 153, 0, 0.4);
    }
    
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #232F3E;
    }
    
    [data-testid="stMetricLabel"] {
        color: #666;
        font-weight: 500;
    }
    
    /* Cards and Containers with hover effects */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #F3F3F3 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #FF9900;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 16px rgba(255, 153, 0, 0.2);
        border-left-width: 6px;
    }
    
    /* Metric hover effects */
    [data-testid="stMetricValue"] {
        transition: all 0.3s ease;
    }
    
    [data-testid="stMetricContainer"]:hover [data-testid="stMetricValue"] {
        transform: scale(1.1);
        color: #FF9900;
    }
    
    /* Text Input Styling with hover effects */
    .stTextArea textarea {
        border-radius: 8px;
        border: 2px solid #E6E6E6;
        transition: all 0.3s ease;
        background: white;
    }
    
    .stTextArea textarea:hover {
        border-color: #FF9900;
        box-shadow: 0 2px 8px rgba(255, 153, 0, 0.1);
    }
    
    .stTextArea textarea:focus {
        border-color: #FF9900;
        box-shadow: 0 0 0 3px rgba(255, 153, 0, 0.2);
        outline: none;
    }
    
    /* Selectbox and Radio Styling with hover effects */
    .stSelectbox label, .stRadio label {
        color: #232F3E;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stSelectbox [data-baseweb="select"] {
        transition: all 0.3s ease;
        border-radius: 8px;
    }
    
    .stSelectbox [data-baseweb="select"]:hover {
        border-color: #FF9900;
        box-shadow: 0 2px 8px rgba(255, 153, 0, 0.2);
    }
    
    .stRadio [data-baseweb="radio"] {
        transition: all 0.3s ease;
    }
    
    .stRadio [data-baseweb="radio"]:hover {
        transform: scale(1.1);
    }
    
    .stRadio [data-baseweb="radio"] label {
        cursor: pointer;
        padding: 0.5rem;
        border-radius: 6px;
        transition: all 0.3s ease;
    }
    
    .stRadio [data-baseweb="radio"] label:hover {
        background: rgba(255, 153, 0, 0.1);
        transform: translateX(4px);
    }
    
    /* Success/Error/Info Messages */
    .stSuccess {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 4px solid #28a745;
        border-radius: 8px;
    }
    
    .stError {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left: 4px solid #dc3545;
        border-radius: 8px;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border-left: 4px solid #17a2b8;
        border-radius: 8px;
    }
    
    /* Dataframe Styling with hover effects */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .dataframe:hover {
        box-shadow: 0 4px 16px rgba(255, 153, 0, 0.2);
    }
    
    .dataframe tbody tr {
        transition: all 0.2s ease;
    }
    
    .dataframe tbody tr:hover {
        background: rgba(255, 153, 0, 0.1);
        transform: scale(1.01);
    }
    
    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #FF9900, transparent);
        margin: 2rem 0;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #232F3E;
    }
    
    h2 {
        border-bottom: 3px solid #FF9900;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }
    
    /* Expander Styling with hover effects */
    .streamlit-expanderHeader {
        background: #F3F3F3;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, #FF9900 0%, #FFB84D 100%);
        color: white;
        transform: translateX(4px);
        box-shadow: 0 2px 8px rgba(255, 153, 0, 0.3);
    }
    
    /* Slider Styling with hover effects */
    .stSlider {
        margin: 1rem 0;
    }
    
    .stSlider [data-baseweb="slider"] {
        transition: all 0.3s ease;
    }
    
    .stSlider [data-baseweb="slider"]:hover {
        transform: scale(1.02);
    }
    
    .stSlider [data-baseweb="slider-track"] {
        background: #E6E6E6;
        transition: all 0.3s ease;
    }
    
    .stSlider [data-baseweb="slider-track"]:hover {
        background: #FF9900;
        opacity: 0.3;
    }
    
    .stSlider [data-baseweb="slider-thumb"] {
        background: #FF9900;
        border: 3px solid white;
        box-shadow: 0 2px 8px rgba(255, 153, 0, 0.4);
        transition: all 0.3s ease;
    }
    
    .stSlider [data-baseweb="slider-thumb"]:hover {
        transform: scale(1.2);
        box-shadow: 0 4px 12px rgba(255, 153, 0, 0.6);
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        background: #F3F3F3;
    }
    
    /* Custom badge styling with hover effects */
    .sentiment-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.3s ease;
        cursor: pointer;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .sentiment-badge:hover {
        transform: scale(1.1) translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .sentiment-positive {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
    }
    
    .sentiment-positive:hover {
        background: linear-gradient(135deg, #c3e6cb 0%, #d4edda 100%);
    }
    
    .sentiment-negative {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        color: #721c24;
    }
    
    .sentiment-negative:hover {
        background: linear-gradient(135deg, #f5c6cb 0%, #f8d7da 100%);
    }
    
    .sentiment-neutral {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        color: #856404;
    }
    
    .sentiment-neutral:hover {
        background: linear-gradient(135deg, #ffeaa7 0%, #fff3cd 100%);
    }
    
    /* Sidebar navigation hover effects */
    [data-testid="stSidebar"] .stRadio [data-baseweb="radio"] label {
        transition: all 0.3s ease;
        padding: 0.5rem;
        border-radius: 6px;
        margin: 0.25rem 0;
    }
    
    [data-testid="stSidebar"] .stRadio [data-baseweb="radio"] label:hover {
        background: rgba(255, 153, 0, 0.2);
        transform: translateX(8px);
        padding-left: 1rem;
    }
    
    /* Checkbox hover effects */
    .stCheckbox label {
        transition: all 0.3s ease;
        cursor: pointer;
        padding: 0.25rem;
        border-radius: 4px;
    }
    
    .stCheckbox label:hover {
        background: rgba(255, 153, 0, 0.1);
    }
    
    /* Number input hover effects */
    .stNumberInput input {
        transition: all 0.3s ease;
        border-radius: 6px;
    }
    
    .stNumberInput input:hover {
        border-color: #FF9900;
        box-shadow: 0 2px 8px rgba(255, 153, 0, 0.1);
    }
    
    .stNumberInput input:focus {
        border-color: #FF9900;
        box-shadow: 0 0 0 3px rgba(255, 153, 0, 0.2);
    }
    
    /* Smooth page transitions */
    .main .block-container {
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Chart container hover effects */
    [data-testid="stPlotlyChart"] {
        transition: all 0.3s ease;
    }
    
    [data-testid="stPlotlyChart"]:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 16px rgba(255, 153, 0, 0.15);
    }
    </style>
""", unsafe_allow_html=True)

# Title with Amazon branding
st.title("📊 Amazon Review Sentiment Analysis Dashboard")
st.caption("Powered by AI • Extract Insights from Customer Reviews")
st.markdown("---")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a page",
    ["🏠 Home", "🔍 Sentiment Analysis", "📈 Data Exploration", "⚙️ About"]
)

# Initialize session state
if 'sample_data' not in st.session_state:
    st.session_state.sample_data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'bert_model_path' not in st.session_state:
    st.session_state.bert_model_path = None

# Home Page
if page == "🏠 Home":
    st.header("Welcome to the Amazon Review Sentiment Analysis Dashboard")
    st.markdown("""
    This dashboard provides tools for analyzing Amazon product reviews and performing sentiment analysis.
    
    ### Features:
    - **Sentiment Analysis**: Analyze individual reviews or batch of reviews
    - **Data Exploration**: Load and explore Amazon review datasets
    - **Preprocessing**: See how reviews are cleaned and processed
    
    ### How to Use:
    1. Navigate to **Sentiment Analysis** to analyze individual reviews
    2. Use **Data Exploration** to load and view sample data
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Categories", "34", help="Amazon product categories analyzed")
    with col2:
        st.metric("Sample Reviews", "340,000+", help="Total reviews in dataset")
    with col3:
        st.metric("Sentiment Classes", "3", help="Positive, Negative, Neutral")

# Sentiment Analysis Page
elif page == "🔍 Sentiment Analysis":
    st.header("🔍 Sentiment Analysis")
    
    # Model selection
    st.subheader("🤖 Model Selection")
    
    if TRANSFORMERS_AVAILABLE:
        model_options = {
            "Simple Rule-Based": "simple",
            "BERT (Pretrained)": "bert-pretrained",
            "Twitter RoBERTa": "twitter-roberta"
        }
        
        selected_model_display = st.selectbox(
            "Choose a sentiment analysis model:",
            list(model_options.keys()),
            help="Select the model to use for sentiment analysis. Transformer models provide more accurate results."
        )
        selected_model = model_options[selected_model_display]
        
        # Show model info
        if selected_model == "bert-pretrained":
            st.info("🤖 Using BERT-base-cased pretrained model. For best results, use a fine-tuned model.")
        elif selected_model == "twitter-roberta":
            st.info("🤖 Using Twitter RoBERTa model, optimized for social media and review text.")
        else:
            st.info("📝 Using simple rule-based analysis (fast but less accurate).")
        
        # Fine-tuned model path input (optional)
        if selected_model == "bert-pretrained":
            with st.expander("🔧 Advanced: Use Fine-tuned Model"):
                st.text_input(
                    "Fine-tuned Model Path (optional):",
                    help="Enter the path to your fine-tuned BERT model checkpoint",
                    key="bert_model_path"
                )
    else:
        st.warning("⚠️ Transformers library not available. Install with: `pip install transformers torch`")
        st.info("📝 Using simple rule-based analysis only.")
        selected_model = "simple"
        selected_model_display = "Simple Rule-Based"
    
    st.markdown("---")
    
    analysis_mode = st.radio(
        "Choose analysis mode:",
        ["Single Review", "Batch Analysis"],
        horizontal=True
    )
    
    if analysis_mode == "Single Review":
        st.subheader("Analyze a Single Review")
        
        # Text input
        review_text = st.text_area(
            "Enter your review text:",
            height=150,
            placeholder="Type your product review here..."
        )
        
        if st.button("Analyze Sentiment", type="primary"):
            if review_text:
                with st.spinner(f"Processing review with {selected_model_display}..."):
                    # Preprocess text
                    processed_text = preprocess_text(review_text)
                    
                    # Get sentiment using selected model
                    if selected_model == "bert-pretrained" and st.session_state.bert_model_path:
                        sentiment, confidence = analyze_sentiment(
                            review_text, 
                            model_type="bert-finetuned",
                            model_path=st.session_state.bert_model_path
                        )
                    else:
                        sentiment, confidence = analyze_sentiment(review_text, model_type=selected_model)
                    
                    # Display results
                    st.markdown("### Results")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        sentiment_emoji = {"positive": "😊", "negative": "😞", "neutral": "😐"}
                        sentiment_class = f"sentiment-{sentiment}"
                        st.markdown(f"""
                            <div class="sentiment-badge {sentiment_class}">
                                <strong>Sentiment: {sentiment.upper()}</strong> {sentiment_emoji.get(sentiment, "")}
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.metric("Confidence", f"{confidence:.2%}")
                    
                    with col3:
                        st.metric("Model Used", selected_model_display)
                    
                    # Show processed text with all preprocessing steps from notebook
                    with st.expander("View All Preprocessing Steps (from notebook pipeline)"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**1. Clean Review (Cell 21):**")
                            clean_rev = create_clean_review(review_text)
                            st.text(clean_rev[:200] + "..." if len(clean_rev) > 200 else clean_rev)
                            
                            st.markdown("**2. Lemmatized Review (Cell 24):**")
                            lemmatized = lemmatize_text(clean_rev)
                            st.text(lemmatized[:200] + "..." if len(lemmatized) > 200 else lemmatized)
                        
                        with col2:
                            st.markdown("**3. Tokenized Review (Cell 33):**")
                            tokens = tokenize_review(clean_rev)
                            st.text(str(tokens[:50]) + "..." if len(tokens) > 50 else str(tokens))
                            
                            st.markdown("**4. Clean Title:**")
                            clean_tit = create_clean_title(review_text.split('.')[0] if '.' in review_text else review_text[:50])
                            st.text(clean_tit)
            else:
                st.warning("Please enter a review text to analyze.")
    
    else:  # Batch Analysis
        st.subheader("Batch Analysis")
        
        reviews_input = st.text_area(
            "Enter multiple reviews (one per line):",
            height=200,
            placeholder="Review 1\nReview 2\nReview 3\n..."
        )
        
        if st.button("Analyze Batch", type="primary"):
            if reviews_input:
                reviews_list = [r.strip() for r in reviews_input.split("\n") if r.strip()]
                
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, review in enumerate(reviews_list):
                    status_text.text(f"Processing review {idx + 1} of {len(reviews_list)}...")
                    processed = preprocess_text(review)
                    
                    # Use selected model for batch analysis
                    if selected_model == "bert-pretrained" and st.session_state.bert_model_path:
                        sentiment, confidence = analyze_sentiment(
                            review, 
                            model_type="bert-finetuned",
                            model_path=st.session_state.bert_model_path
                        )
                    else:
                        sentiment, confidence = analyze_sentiment(review, model_type=selected_model)
                    
                    results.append({
                        "Review": review[:100] + "..." if len(review) > 100 else review,
                        "Sentiment": sentiment,
                        "Confidence": f"{confidence:.2%}",
                        "Processed": processed[:50] + "..." if len(processed) > 50 else processed
                    })
                    progress_bar.progress((idx + 1) / len(reviews_list))
                
                progress_bar.empty()
                status_text.empty()
                
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)
                
                # Sentiment distribution
                col1, col2 = st.columns(2)
                
                with col1:
                    sentiment_counts = results_df['Sentiment'].value_counts()
                    fig = px.pie(
                        values=sentiment_counts.values,
                        names=sentiment_counts.index,
                        title="Sentiment Distribution",
                        color_discrete_map={
                            "positive": "#28a745",
                            "negative": "#dc3545",
                            "neutral": "#FF9900"
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Confidence distribution
                    results_df['Confidence_Num'] = results_df['Confidence'].str.rstrip('%').astype('float') / 100
                    avg_confidence = results_df['Confidence_Num'].mean()
                    st.metric("Average Confidence", f"{avg_confidence:.2%}")
                    st.metric("Total Reviews Analyzed", len(results_df))
                    st.metric("Model Used", selected_model_display)
            else:
                st.warning("Please enter at least one review to analyze.")

# Data Exploration Page
elif page == "📈 Data Exploration":
    st.header("📈 Data Exploration")
    
    st.info("💡 Note: Loading full dataset requires HuggingFace access. Use sample data for demonstration.")
    
    load_option = st.radio(
        "Choose data source:",
        ["Load Sample Data", "Load from HuggingFace (Full Dataset)"],
        horizontal=True
    )
    
    if load_option == "Load Sample Data":
        if st.button("Load Sample Reviews", type="primary"):
            with st.spinner("Loading sample data..."):
                sample_data = load_sample_data()
                if sample_data is not None and len(sample_data) > 0:
                    st.session_state.sample_data = sample_data
                    st.success(f"✅ Loaded {len(sample_data)} reviews!")
                else:
                    st.error("Failed to load sample data.")
        
        if st.session_state.sample_data is not None:
            df = st.session_state.sample_data
            
            st.subheader("Dataset Overview")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Reviews", f"{len(df):,}")
            with col2:
                st.metric("Categories", df['category'].nunique() if 'category' in df.columns else 0)
            with col3:
                st.metric("Avg Rating", f"{df['rating'].mean():.2f}" if 'rating' in df.columns else "N/A")
            with col4:
                st.metric("Unique Products", df['asin'].nunique() if 'asin' in df.columns else "N/A")
            
            # Filter options
            st.subheader("Filters")
            col1, col2 = st.columns(2)
            with col1:
                if 'category' in df.columns:
                    categories = ['All'] + sorted(df['category'].unique().tolist())
                    selected_category = st.selectbox("Category", categories)
            with col2:
                if 'rating' in df.columns:
                    min_rating, max_rating = int(df['rating'].min()), int(df['rating'].max())
                    rating_range = st.slider("Rating Range", min_rating, max_rating, (min_rating, max_rating))
            
            # Apply filters
            filtered_df = df.copy()
            if 'category' in df.columns and selected_category != 'All':
                filtered_df = filtered_df[filtered_df['category'] == selected_category]
            if 'rating' in df.columns:
                filtered_df = filtered_df[
                    (filtered_df['rating'] >= rating_range[0]) & 
                    (filtered_df['rating'] <= rating_range[1])
                ]
            
            st.subheader("Review Data")
            st.dataframe(
                filtered_df[['rating', 'title', 'text', 'category']].head(100),
                use_container_width=True,
                height=400
            )
            
            # Show sample review details
            if len(filtered_df) > 0:
                st.subheader("Sample Review Details")
                review_idx = st.slider("Select Review Index", 0, min(50, len(filtered_df)-1), 0)
                selected_review = filtered_df.iloc[review_idx]
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.metric("Rating", f"{selected_review['rating']} ⭐")
                    if 'category' in selected_review:
                        st.write(f"**Category:** {selected_review['category']}")
                with col2:
                    st.write("**Title:**", selected_review.get('title', 'N/A'))
                    st.write("**Review Text:**")
                    st.text_area("", selected_review.get('text', 'N/A'), height=150, disabled=True)
    
    else:  # Load from HuggingFace
        st.warning("⚠️ Full dataset loading requires HuggingFace authentication and may take time.")
        
        col1, col2 = st.columns(2)
        with col1:
            selected_categories = st.multiselect(
                "Select Categories",
                ["Software", "Video_Games", "All_Beauty", "Electronics", "Books"],
                default=["Software"]
            )
        with col2:
            n_reviews = st.number_input("Number of Reviews per Category", 100, 10000, 1000, 100)
        
        if st.button("Load from HuggingFace", type="primary"):
            st.info("This feature requires HuggingFace setup. Please use sample data for now.")

# About Page
elif page == "⚙️ About":
    st.header("⚙️ About This Dashboard")
    
    st.markdown("""
    ### Project Overview
    This dashboard is part of the **Cadence Design Systems 2A AI Studio Project**.
    
    The goal of this project is to develop an AI system that automatically:
    - Extracts product features from Amazon product reviews
    - Performs sentiment analysis for each feature
    - Prioritizes issues based on frequency and severity
    - Generates recommendations for product development
    
    ### Features
    - **Sentiment Analysis**: Classify reviews as positive, negative, or neutral
    - **Data Preprocessing**: Text normalization, lemmatization, and tokenization
    - **Visualization**: Interactive charts and statistics
    - **Data Exploration**: Load and explore Amazon review datasets
    
    ### Team Members
    - Dhruhi Sheth
    - Lisa Yu
    - Miller Liu
    - Yong Thu La Wong
    - Tara Rezaei
    - Raghav Sriram
    - Kyi Lei Aye
    
    ### Data Source
    The dataset used is the [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) 
    from HuggingFace, containing reviews across 34 product categories.
    
    ### Milestones
    - **Milestone 1**: Data Exploration & Preprocessing ✅
    - **Milestone 2**: Model Development & Deployment 🚧
    - **Milestone 3**: Finalization & Presentation 📋
    """)
    
    st.markdown("---")
    st.markdown("**Built with Streamlit** 🚀")

