# Portfolio
Welcome to my project portfolio! This repository showcases my skills in data analysis, modeling, machine learning, and deep learning.
## 📈 Time Series Forecasting
**Problem Statement**:  
Forecast flight delays for the next 12 time periods using historical flight data.

**Data**:  
- Source: Provided delayed flight dataset [`DelayedFlights.csv`](https://www.kaggle.com/datasets/giovamata/airlinedelaycauses)
- Description: Includes flight numbers, departure/arrival delays, scheduled and actual times.

**Data Mining Operations**:  
- Data wrangling: Cleaned missing values, transformed date-time fields, indexed by time.
- Modeling: Built an ARIMA model after analyzing ACF/PACF plots and differencing for stationarity.
- Libraries: `pandas`, `matplotlib`, `statsmodels`
- Reason for model choice: ARIMA is effective for time series forecasting with autocorrelation patterns.

**Model Outputs**:  
- Visualized trends and seasonality.
- Generated a 12-step ahead forecast.
- Evaluated performance using MAE and RMSE.

**Limitations**:  
- Limited data history may impact long-term forecast stability.
- External factors affecting flight delays (e.g., weather) are not included.

**Were you able to effectively solve the problem?**  
Yes, the ARIMA model produced reasonable and statistically evaluated forecasts for future periods.

[Time-Series-Forecasting](https://github.com/HenryRomeo1/Time-Series-Forecasting/tree/main)

---

## 💬 Sentiment and Emotion Analysis 
**Problem Statement**:  
Analyze airline customer tweets to determine the overall sentiment and emotional tone.

**Data**:  
- Source: Airline Twitter Sentiment dataset [`Tweets.csv`](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
- Description: Customer tweets labeled with positive, neutral, or negative sentiment.

**Data Mining Operations**:  
- Data wrangling: Cleaned text data (removed mentions, URLs, punctuation).
- Modeling: Applied VADER SentimentIntensityAnalyzer for sentiment classification. Used NRCLex for emotion tagging.
- Libraries: `nltk`, `matplotlib`, `wordcloud`, `nrclex`
- Reason for model choice: VADER is optimized for social media sentiment; NRCLex provides rich emotion detection.

**Model Outputs**:  
- Word Cloud highlighting frequent terms.
- Sentiment Distribution bar chart.
- Emotion Distribution bar chart (fear, sadness, anger, joy, surprise).

**Limitations**:  
- Sarcasm and mixed emotions in tweets can lead to misclassification.
- NRCLex detects a wide range of emotions that had to be filtered for analysis.

**Were you able to effectively solve the problem?**  
Yes, sentiment and emotion patterns were effectively extracted, providing insights into airline customer experiences.


[Sentiment and Emotion Analysis](https://github.com/HenryRomeo1/Sentiment-and-Emotion-Analysis-of-Airline-Tweets/tree/main)

---
## 🐱🐶 Deep Learning: Image Classification (Cats vs Dogs)

**Problem Statement**:  
Classify images as cat or dog using a deep learning convolutional neural network.

**Data**:  
- Source: TensorFlow Datasets [`cats_vs_dogs`](https://www.tensorflow.org/datasets/catalog/cats_vs_dogs)
- Description: 25,000 labeled images of cats and dogs.

**Data Mining Operations**:  
- Data wrangling: Resized and normalized image data.
- Modeling: Used MobileNetV2 pre-trained CNN for transfer learning.
- Libraries: `tensorflow`, `keras`, `numpy`
- Reason for model choice: MobileNetV2 provides high accuracy with efficient computational resources.

**Model Outputs**:  
- Achieved 98% validation accuracy.
- Plotted training and validation loss/accuracy curves.

**Limitations**:  
- Highly curated dataset may not generalize perfectly to wild images.
- Limited augmentation might underprepare model for unseen conditions.

**Were you able to effectively solve the problem?**  
Yes, the model achieved very high accuracy on test data, successfully solving the classification task.

[Deep Learning: Image Classification](https://github.com/HenryRomeo1/Deep-Learning-Image-Classification)

---

## 📚 Text Classification Project for 10,000 Rotten Tomato Reviews

**Problem Statement**:  
Classify text documents based on sentiment or topic using machine learning.

**Data**:  
- Source: Source: Hugging Face dataset [Rotten Tomatoes](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes)
- Description: Documents labeled into different categories.

**Data Mining Operations**:  
- NLP preprocessing (lowercase, tokenization, stopword removal).
- Built a machine learning model for text classification (likely Logistic Regression or Neural Network).
- Libraries: `tensorflow`, `keras`, `scikit-learn`
- Reason for model choice: Effective balance between performance and explainability.

**Model Outputs**:  
- Achieved classification performance metrics (accuracy, precision, recall).

**Limitations**:  
- Dataset size and vocabulary variability may impact model performance.

**Were you able to effectively solve the problem?**  
Yes, text classification accuracy met expectations for the dataset provided.

[Text Classification](https://github.com/HenryRomeo1/Text-Classification/tree/main)

---

## 📈 Linear Regression (Simple and Multivariate)

**Problem Statement**:  
Predict a continuous dependent variable based on one or more independent variables.

**Data**:  
- Source: Simple regression (`Pearson`).
- Description: CSV files containing independent and dependent variable data.

**Data Mining Operations**:  
- Loaded data, explored initial distributions.
- Conducted simple and multiple linear regression models.
- Preprocessed data (normalization using Min-Max Scaling).
- Generated correlation matrix and heatmaps.
- Split dataset into training (80%) and testing (20%) sets.
- Trained and evaluated model using performance metrics.

**Model Outputs**:  
- Regression plots.
- Correlation heatmaps.
- Evaluation metrics (MSE, R² Score).

**Limitations**:  
- Small dataset sizes could introduce variance in performance metrics.

**Were you able to effectively solve the problem?**  
Yes, models achieved reasonable fits, and predictions aligned with expected trends.

[Linear Regression]([HW4_Part1_Henry_Romero.ipynb](https://github.com/HenryRomeo1/Linear-Regression-))

---

## 📈 Multivaritive Linear Regression

**Problem Statement**:  
Apply multivariate linear regression on extended datasets.

**Data**:  
- Source: Source: Simple regression (`advertising`).

**Data Mining Operations**:  
- Handled additional features and multicollinearity checks.
- Re-trained models with multiple predictors.
- Refined model with feature selection and engineering.

**Model Outputs**:  
- Improved regression graphs and prediction accuracy.
- Updated correlation matrices.

**Limitations**:  
- Adding irrelevant variables could increase noise in model results.

**Were you able to effectively solve the problem?**  
Yes, multivariate regressions demonstrated improved predictive performance.

[Multivaritive Linear Regression](https://github.com/HenryRomeo1/Regression)

---

## 📉 Logarithmic Regression

**Problem Statement**:  
Model relationships between variables using a logarithmic regression approach.

**Data**:  
- Source: Logarithmic Regressions Dataset

**Data Mining Operations**:  
- Conducted logarithmic transformations on independent variables.
- Applied regression modeling post-transformation.
- Split dataset into training/testing.
- Evaluated model performance with standard regression metrics.

**Model Outputs**:  
- Log-transformed regression plots.
- Performance evaluation metrics.

**Limitations**:  
- Logarithmic transformations may not be appropriate for all variables.

**Were you able to effectively solve the problem?**  
Yes, the logarithmic model better captured nonlinear relationships compared to simple linear models.

[Logarithmic Regression](https://github.com/HenryRomeo1/Logarithmic-Regression/tree/main)

---

## 🤖 k-Nearest Neighbors (kNN) Classification

**Problem Statement**:  
Classify instances into categories based on proximity to training samples using the kNN algorithm.

**Data**:  
- Source: kNN Classification Dataset (`bank`)

**Data Mining Operations**:  
- Performed normalization.
- Applied kNN classification.
- Created confusion matrix and calculated accuracy score.

**Model Outputs**:  
- Confusion matrix visualization.
- Accuracy, precision, and recall scores.

**Limitations**:  
- Sensitivity to choice of `k` and feature scaling.

**Were you able to effectively solve the problem?**  
Yes, the model achieved good accuracy, showing strong classification performance.

[KNN Classification](https://github.com/HenryRomeo1/KNN-Classification)

---

## 🤖 Naive Bayes Classification

**Problem Statement**:  
Classify observations using Gaussian Naive Bayes algorithm based on probability theory.

**Data**:  
- Source: Naive Bayes Dataset (`Naive-Bayes-Classification-Data`)

**Data Mining Operations**:  
- Applied Gaussian Naive Bayes classification.
- Built confusion matrix and computed accuracy.
- Evaluated model performance using cross-tabulations.

**Model Outputs**:  
- Prediction labels.
- Model evaluation metrics.

**Limitations**:  
- Assumes feature independence, which may not always hold.

**Were you able to effectively solve the problem?**  
Yes, the Naive Bayes classifier produced high accuracy, particularly for simpler feature sets.

[Naive Bayes Classification](https://github.com/HenryRomeo1/Naive-Bayes-Classification)

---




