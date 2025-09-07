# ⚡ Sentiment Analysis on Amazon Reviews  

![Python](https://www.google.com/url?sa=i&url=https%3A%2F%2Frealpython.com%2Fpython-nltk-sentiment-analysis%2F&psig=AOvVaw0dEIKZMo80ScbCuENoQvHY&ust=1757359537243000&source=images&cd=vfe&opi=89978449&ved=0CBYQjRxqFwoTCJj72tCwx48DFQAAAAAdAAAAABAE)

A complete **Natural Language Processing (NLP)** project in Python to analyse the sentiment of Amazon Fine Food Reviews using both **lexicon-based (VADER)** and **transformer-based (RoBERTa)** models.  

---

## 📖 Project Overview  
This project demonstrates two approaches to **sentiment analysis**:  

1. **VADER (Valence Aware Dictionary for Sentiment Reasoning)** – a lexicon and rule-based sentiment analysis tool.  
2. **RoBERTa Transformer (via Hugging Face)** – a pre-trained deep learning model capable of contextual sentiment understanding.  

It also explores Hugging Face’s **pipeline API** for production-ready predictions.  

---

## 🛠️ Tech Stack  
- 🐍 **Python**  
- 📂 **pandas, numpy** – Data handling  
- 📊 **matplotlib, seaborn** – Data visualization  
- 🔠 **NLTK** – Tokenization, POS tagging, VADER sentiment  
- 🤖 **Hugging Face Transformers** – RoBERTa model + pipeline API  
- ⚡ **tqdm** – Progress bar for loops  
- 🧮 **scipy** – Softmax for probability conversion  

---

## 📁 Dataset  
- **Source**: [Amazon Fine Food Reviews](https://www.kaggle.com/code/robikscube/sentiment-analysis-python-youtube-tutorial/input)  
- **Description**: ~500,000 food product reviews including rating (1–5 stars), text, and metadata.  
- **Subset Used**: 500 reviews for demonstration (scalable to full dataset).  
- **Key Attributes**:  
  - Review ID  
  - Product ID  
  - User ID  
  - Review Text  
  - Score (1–5 stars)  

---

## 🚀 Features & Insights  

### Business Problem  
Understanding customer sentiment at scale is critical for **product development, brand reputation, and customer satisfaction analysis**. Manual review reading is impractical at this size.  

### Solution: Dual-Model Sentiment Analysis  
This project implements both **lexicon-based** and **transformer-based** models to analyze text sentiment and compare performance.  

---

### 🔑 Key Steps & Visuals  

#### 1. Exploratory Data Analysis  
- Distribution of review scores.  
- Data skew identified (majority 5-star reviews).  

#### 2. VADER Model (Lexicon-Based)  
- Outputs **positive, neutral, negative, compound scores**.  
- Validated against review star ratings.  
- Visualized correlation between sentiment and ratings.  

#### 3. RoBERTa Model (Transformer-Based)  
- Pre-trained on large-scale text data.  
- Handles context and sarcasm better than VADER.  
- Applied across dataset with error handling.  

#### 4. Model Comparison  
- **Pairplots** and metrics comparing VADER vs. RoBERTa.  
- Case studies of misaligned predictions.  

#### 5. Hugging Face Pipeline  
- Two-line implementation for production:  
```python
from transformers import pipeline
sentiment_pipeline = pipeline("sentiment-analysis")
sentiment_pipeline("The food was amazing!")
