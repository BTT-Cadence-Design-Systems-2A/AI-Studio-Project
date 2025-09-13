# Cadence-Design-Systems-2A-AI-Studio-Project
#### The goal of this project is to develop an AI system that automatically extracts product features from [Amazon large-scale product review dataset](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023), performs sentiment analysis for each feature, prioritizes issues based on their frequency and severity and generates recommendations for which features to preserve and which to redesign. Building on this analysis, we will create an AI-powered dashboard that enables product teams to efficiently extract actionable insights from thousands of customer reviews, providing clear guidance for product development and improvement.

Team Members
- Dhruhi Sheth
- Lisa Yu
- Miller Liu
- Yong Thu La Wong
- Tara Rezaei
- Raghav Sriram
- Kyi Lei Aye

## Milestone 1. Data Exploration & Preprocessing
### Data Loading 
There are 34 product categories, and each category contains two JSON files: one for reviews and one for metadata. Using the fsspec library, these JSON files were loaded from HuggingFace and read line by line. With the Pandas library, two DataFrames were created—one for reviews and one for metadata. All review files were consolidated into a single review DataFrame, and all metadata files into a single metadata DataFrame. The two DataFrames were then merged on the asin and parent_asin columns. This combined dataset enables retrieval of customer review content and supports subsequent data cleaning processes. 

### Data Cleaning
Since our first task is doing sentiment analysis on customer feedback review data to classify the good feedback or bad feedback, the relevant columns (rating, title, text, category) in the dataset were extracted. Then, for each column, we checked null values and remove those.

### Data Transformation
#### Text Normalization 
We removed punctuation from the review text and created a new column containing normalized text reviews.

#### Lemmatization 
Text reviews were lemmatized using the WordNetLemmatizer provided by the Natural Language Toolkit (NLTK).

#### Mapping Rating Stars to Sentiment Labels
We assigned sentiment labels according to the rating stars: ratings 1–2 were mapped to “negative,” 3 to “neutral,” and 4–5 to “positive.”

#### Text Tokenization
The text reviews were tokenized into individual words using the word_tokenize function from the Natural Language Toolkit (NLTK).

## Milestone 2. Model Development & Deployment
## Milestone 3. Finalization & Presentation
