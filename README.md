# text-classification
# Sentiment Analysis on IMDB Reviews 🎬📊

## 📌 Project Overview
This project focuses on **text classification** using Natural Language Processing (NLP).  
We analyze movie reviews from the **IMDB Dataset** and classify them as **Positive** or **Negative**.  

The pipeline covers **data preprocessing, feature engineering, and machine learning model building**, showcasing multiple NLP techniques.

---

## 🚀 Features
- **Preprocessing**: HTML tag removal, lowercasing, stopword removal.  
- **Feature Extraction**:  
  - Bag-of-Words (BOW)  
  - TF-IDF  
  - Word2Vec embeddings (Gensim)  
- **Machine Learning Models**:  
  - Naïve Bayes  
  - Random Forest Classifier  
- **Performance**: Achieved **~84% accuracy** using n-gram + Random Forest.  

---

## 📂 Dataset
- **IMDB Dataset of 50K Movie Reviews**  
- For this project, ~16k samples were used for training and evaluation.  

---

## 🛠️ Tech Stack
- **Python**  
- **Libraries**: NumPy, Pandas, Scikit-learn, NLTK, Gensim, Matplotlib  

---

## 📊 Results
| Feature Extraction | Model              | Accuracy |
|--------------------|--------------------|----------|
| Bag-of-Words       | GaussianNB         | 64%      |
| Bag-of-Words (n-grams) | Random Forest  | **84%**  |
| TF-IDF             | Random Forest      | ~84%     |
| Word2Vec           | Random Forest      | ~83%     |


📈 Future Improvements

1. Implement deep learning models (LSTM, BiLSTM, Transformers).

2. Hyperparameter tuning for Random Forest & TF-IDF.

3. Deploy model as a simple Flask/Streamlit web app.   
  
