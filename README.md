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
### Data Loading and Cleaning
#### There are 34 product categories, and each category contains two JSON files: one for reviews and one for metadata. Using the fsspec library, these JSON files were loaded from HuggingFace and read line by line. With the Pandas library, two DataFrames were created—one for reviews and one for metadata. All review files were consolidated into a single review DataFrame, and all metadata files into a single metadata DataFrame. The two DataFrames were then merged on the asin and parent_asin columns. This combined dataset enables retrieval of customer review content and supports subsequent data cleaning processes.

### Data Transformation
#### Normalize text (lowercasing, lemmatization) and map star ratings to sentiment labels etc. …

### Data Reduction 
#### Filter reviews by category or product type to reduce dataset size for faster iteration …

### Data Augmentation 
#### Use paraphrasing or back-translation to increase training data diversity for sentiment classification … 


## Milestone 2. Model Development & Deployment
## Milestone 3. Finalization & Presentation
