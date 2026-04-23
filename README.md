# ✈️ Airline Tweet Sentiment Analysis

> An end-to-end NLP & Machine Learning project that classifies customer sentiment from Twitter data across 6 major US airlines.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.7.2-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-2.3.3-lightblue?logo=pandas)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## Overview

This project performs **multi-class sentiment classification** (Positive / Neutral / Negative) on **14,640 tweets** directed at major US airlines, collected in February 2015. It walks through the complete data science pipeline — from raw data exploration to a live prediction interface.

---

## 📁 Project Structure

```
airline-sentiment-analysis/
│
├── Analysis_data.ipynb       # Main Jupyter notebook (full pipeline)
├── plots/                    # Auto-generated EDA & model charts (13 total)
├── data/
│   └── tweets.csv            # Dataset (download from Kaggle)
└── README.md
```



## Pipeline Breakdown

| Step | Description |
|------|-------------|
| **1. Setup & Imports** | Load all libraries, verify environment |
| **2. Load & Explore Data** | Preview dataset, inspect columns and data types |
| **3. Data Cleaning** | Remove mentions, URLs, hashtags, stopwords; normalize text |
| **4. EDA** | Sentiment distribution, per-airline breakdown, negative reason analysis |
| **5. Word Cloud Analysis** | Frequency maps for each sentiment class |
| **6. Time Series Analysis** | Tweet volume trends across the collection period |
| **7. ML Modeling** | Train Naive Bayes, Logistic Regression, and Random Forest |
| **8. Model Evaluation** | Accuracy, confusion matrix, classification report, cross-validation |
| **9. Predict New Tweets** | Live sentiment prediction with confidence scores |
| **10. Conclusions** | Key takeaways and insights |


## Results

| Metric | Value |
|--------|-------|
| **Best Model** | Logistic Regression |
| **Test Accuracy** | ~79% |
| **Total Tweets** | 14,640 |
| **Airlines Covered** | 6 |
| **Sentiment Classes** | Positive · Neutral · Negative |
| **Total Charts Generated** | 13 |



## ✈️ Airlines Analyzed

- United Airlines
- Delta Air Lines
- Southwest Airlines
- American Airlines
- US Airways
- Virgin America


## Tech Stack

| Category | Libraries |
|----------|-----------|
| **Data** | `pandas`, `numpy` |
| **Visualization** | `matplotlib`, `seaborn`, `plotly` |
| **NLP** | `wordcloud`, `sklearn TfidfVectorizer` |
| **ML Models** | `MultinomialNB`, `LogisticRegression`, `RandomForestClassifier` |
| **Evaluation** | `classification_report`, `confusion_matrix`, `cross_val_score` |

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/your-username/airline-sentiment-analysis.git
cd airline-sentiment-analysis
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn plotly wordcloud scikit-learn
```

### 3. Download the dataset

Get the dataset from Kaggle:  
[US Airline Twitter Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)

Place `tweets.csv` inside the `data/` folder.

### 4. Run the notebook

```bash
jupyter notebook Analysis_data.ipynb
```

---

##  Sample Predictions

```
Tweet     : @Delta thanks for the smooth flight and amazing crew!
Predicted : ✅ POSITIVE  (Confidence: 84.5%)

Tweet     : @United my flight was delayed 4 hours. Terrible!
Predicted : ❌ NEGATIVE  (Confidence: 99.3%)

Tweet     : @SouthwestAir is the flight on time today?
Predicted : 😐 NEUTRAL   (Confidence: 62.7%)
```

---

## Key Insights

- **~63% of tweets are negative**, reflecting widespread dissatisfaction with airline service
- **United Airlines** received the highest volume of negative tweets
- Top negative reasons: *Customer Service Issues*, *Late Flights*, *Can't Tell*
- **Logistic Regression** with TF-IDF features outperforms Naive Bayes and Random Forest
- Word clouds reveal stark vocabulary differences between sentiment classes



## Dataset

**Source:** [Crowdflower / Figure Eight — Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)  
**Size:** 14,640 tweets · 15 columns  
**Period:** February 2015

