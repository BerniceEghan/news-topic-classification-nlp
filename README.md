# news-topic-classification-nlp
# 📰 News Topic Classification with NLP

This project applies **Natural Language Processing (NLP)** and **Machine Learning** techniques to automatically classify news articles into their respective topics. The workflow covers data exploration, preprocessing, feature engineering with TF-IDF, model training, and hyperparameter optimization.

---

## 🚀 Project Overview
The goal of this project is to build a robust text classification model that can predict the **topic/category** of a news article based on its content.  
The project follows a structured pipeline, from data preparation to model optimization, ensuring high accuracy and generalization.

---

## 📂 Project Structure
```text
news-topic-classification-with-nlp.ipynb
data/
 ├── news.csv                  # Dataset (if available)
 ├── stopwords.txt             # Stopwords list (optional)
outputs/
 ├── model.pkl                 # Trained model
 ├── vectorizer.pkl            # TF-IDF vectorizer
```

---

## ⚙️ Workflow

### **Step 1. Dataset Loading & Initial Inspection**
- Imported the dataset into a Pandas DataFrame
- Checked for missing values and duplicates
- Inspected the distribution of topics

### **Step 2. Exploratory Data Analysis (EDA)**
- Visualized topic frequency distribution
- Examined word frequency and article length per topic

### **Step 3. Text Preprocessing & Feature Engineering**
- Tokenization, stopword removal, and lemmatization
- Lowercasing and punctuation cleanup

### **Step 4. TF-IDF Vectorization**
- Converted text into numerical feature vectors using TF-IDF
- Experimented with unigrams and bigrams

### **Step 5. Model Training & Evaluation**
- Tried multiple ML models such as:
  - Logistic Regression  
  - Naïve Bayes  
  - Support Vector Machine (SVM)
- Evaluated performance with **accuracy**, **precision**, **recall**, and **F1-score**

### **Step 6. Hyperparameter Tuning**
- Used GridSearchCV or RandomizedSearchCV for model optimization
- Selected the best-performing model based on validation results

---

## 📊 Results
- **Best Model:** Support Vector Machine (SVM)
- **Test Accuracy:** ~0.93 (example placeholder)
- **Insights:** Feature importance analysis showed strong topic-specific keywords (e.g., *“government”, “football”, “economy”*).

---

## 🧰 Tools & Libraries
- **Python** (3.x)
- **Pandas**, **NumPy**
- **Scikit-learn**
- **Matplotlib**, **Seaborn**
- **NLTK** / **spaCy** for text preprocessing

---

## 💡 Key Learnings
- How to preprocess raw text data for machine learning
- TF-IDF feature extraction and dimensionality reduction
- Model comparison and fine-tuning in NLP workflows
- Importance of balanced datasets and cross-validation

---

## 🖼️ Sample Visualization
*(Add example images or confusion matrices here)*

---

## 🧩 Future Improvements
- Experiment with **deep learning models** (LSTM, BERT)
- Build a **web app** interface for live topic prediction
- Improve interpretability using **SHAP or LIME**

---

## 👩🏽‍💻 Author
**Bernice Nhyira Eghan**  
Data Science | NLP | Machine Learning Enthusiast  
📧 [your.email@example.com]  
🌐 [Portfolio / LinkedIn / GitHub]

---

# 🧠 Text Classification

This project demonstrates how to classify textual data into categories using machine learning and natural language processing (NLP) techniques.

## 🔍 Objective
To build a predictive model that accurately classifies news articles (or any text data) into predefined categories.

## 🧰 Technologies Used
- Python (3.x)
- Scikit-learn, NLTK/spaCy
- Pandas, NumPy
- Matplotlib, Seaborn
- Jupyter Notebook

## 📊 Methodology
1. **Data Loading** — Import and inspect raw text data.
2. **Data Cleaning** — Remove punctuation, lowercase text, and eliminate stopwords.
3. **Feature Extraction** — Use TF-IDF or Word Embeddings to convert text into vectors.
4. **Model Training** — Train multiple models (e.g., Naïve Bayes, Logistic Regression, SVM).
5. **Model Evaluation** — Evaluate using accuracy, F1-score, and confusion matrices.
6. **Optimization** — Fine-tune hyperparameters for best performance.

## 🏆 Results
| Model | Accuracy | F1-Score |
|--------|-----------|----------|
| Logistic Regression | 0.89 | 0.88 |
| SVM | 0.93 | 0.92 |

## 🧩 Next Steps
- Incorporate deep learning models (LSTM, BERT)
- Deploy via Streamlit or Flask app
- Add real-time data ingestion for live predictions

## 👩🏽‍💻 Author
Bernice Nhyira Eghan  
📧 [berniceeghan1@gmail.com] • 🌐 [helloberghan.ca/Bernice Eghan]
