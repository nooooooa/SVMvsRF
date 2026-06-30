# Restaurant Performance Analysis Using Sentiment Analysis

This project analyzes restaurant customer reviews using Natural Language Processing (NLP) and Machine Learning to classify review sentiment. The dataset is preprocessed, transformed into TF-IDF features, and evaluated using two classification models: Support Vector Machine (SVM) and Random Forest.

## Project Objectives

- Analyze restaurant customer reviews.
- Perform text preprocessing for NLP tasks.
- Classify reviews into **Positive**, **Neutral**, and **Negative** sentiments.
- Compare the performance of SVM and Random Forest classifiers.
- Visualize sentiment distribution and model performance.

## Dataset

The project uses the **Yelp Review Dataset**.

Each review is converted into sentiment labels based on its star rating:

| Stars | Sentiment |
|-------:|-----------|
| 1–2 | Negative |
| 3 | Neutral |
| 4–5 | Positive |

## Workflow

1. Load and inspect the dataset.
2. Perform text preprocessing:
   - Lowercasing
   - Remove punctuation and numbers
   - Tokenization
   - Stopword removal
   - Stemming
3. Explore the dataset with visualizations.
4. Generate Word Clouds for each sentiment class.
5. Convert text into numerical features using **TF-IDF**.
6. Split data into training (80%) and testing (20%).
7. Train two machine learning models:
   - Linear Support Vector Machine (LinearSVC)
   - Random Forest Classifier
8. Evaluate models using:
   - Accuracy
   - Precision
   - Recall
   - F1-Score
   - Confusion Matrix
9. Compare model performance and identify important features.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- NLTK
- Scikit-learn
- WordCloud

## Visualizations

The notebook includes:

- Star Rating Distribution
- Review Length Distribution
- Sentiment Distribution
- Word Clouds by Sentiment
- Model Performance Comparison
- Confusion Matrices
- SVM Top Features
- Random Forest Feature Importance

## Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Classification Report
- Confusion Matrix

## Project Structure

```
.
├── restaurant_sentiment_analysis.ipynb
├── confusion_matrices.png
├── rf_feature_importance.png
└── README.md
```

## How to Run

1. Clone this repository.

```bash
git clone https://github.com/your-username/your-repository.git
```

2. Install the required packages.

```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn wordcloud
```

3. Download the required NLTK resources.

```python
import nltk
nltk.download("punkt")
nltk.download("stopwords")
```

4. Open and run:

```
restaurant_sentiment_analysis.ipynb
```

## Results

The project compares the performance of **Linear SVM** and **Random Forest** for restaurant review sentiment classification. Their performance is evaluated using multiple classification metrics and visualization techniques to determine the most effective model for the dataset.

## License

This project is intended for educational and research purposes.
