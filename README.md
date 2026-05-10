# Tech-Job-Recommendation-System

We recommend reading our files in this order:

1. DataClean_TechJobMarketSkills.ipynb
- There were two attempts of making an ideal data set. This creates "Tech_Market_CleanSkills.csv". We ultimately picked "merged ai and job maket.csv" (code found in KClustering.ipynb)

2. regression_model.py

3. Recommendations_w_TF-IDF.ipynb
   - This notebook includes the tfidf.py file as we use its mechanism to test its output. Here, we created our "Recommendation Engine," which is what we coined our mechanism for users to input their skills and receive a job recommendation. 
5. KClustering.ipynb
6. Final_nn_model1_model2_RecEngine + tfidf_nn_model.ipynb
   - Final_nn_model1_model2_RecEngine is a notebook that includes the model from tfidf_nn_model.ipynb. In Final_nn_model1_model2_RecEngine, we implement our Recommendation engine that integrates predictions from the FNN and TF-IDF.
  
Notes:

Folder called "raw data" includes the raw data and output files from our data cleaning. 

# About

Recommends relevant job postings based on a user's skills and experience level using TF-IDF vectorization and cosine similarity
Predicts estimated salary from skills and experience using a feedforward neural network built in PyTorch

Stack: Python, PyTorch, scikit-learn, pandas, numpy
Models: TF-IDF + Cosine Similarity for recommendations, Feedforward Neural Network for salary prediction
Dataset: 11,500+ job postings merged from two sources — a structured AI job market dataset and a real-world job listings dataset

Kaggle:
https://www.kaggle.com/datasets/sethelm/tech-job-market-insights-march-2026-edition  
https://www.kaggle.com/datasets/shree0910/ai-and-data-science-job-market-dataset-20202026 
