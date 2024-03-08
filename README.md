
# Problem Statement:
## Introduction:
This project aims to create a system that uses Natural Language Processing (NLP) to organize Reddit posts into different categories. Specifically, it focuses on sorting posts from subreddits about investing and stocks.

In today's digital era, social media platforms serve as vibrant hubs for a multitude of communities, often forming implicit 'tribes'. Recognizing and comprehending these communities holds paramount importance for various purposes, ranging from enhancing organizational efficiency to bolstering revenue generation.

## Objects:

### Primary goal:
Developing a classifier to predict the sentiment of posts on the subreddits r/investing, and r/stocks. The classifier will accurately categorize posts into positive, negative, or neutral sentiment categories based on the content of the posts. This will assist users in quickly identifying the prevailing sentiment within these investment-related communities, aiding in decision-making processes related to stock market investments and analysis.

## Methodology:
### Approach
Using NLP methods to handle and examine textual data from subreddits r/investing and r/stocks. Employing sentiment analysis algorithms to identify the predominant emotional undertones within these two groups.

### 1. Data Collection
- Data was collected using a Reddit API to scrape posts from subsequent subreddits:
  <ul>
  <li> Investing </li>
  <li> Stocks </li>
  </ul>
- I gathered data from Reddit's API using PRAW for 5 continuous days.
- The data from each subreddit was saved separately.

### 2. Data Cleaning and Exploratory Data Analysis
- The data cleaning process:
  <ul>
  <li> Null values were removed.</li>
  <li> Feature engineering was performed.</li>
  <li>Missing values were imputed.</li>
  <li>Data was normalized and encoded.</li>
  <li>A secondary check for duplicates was conducted to ensure the absence of duplicates in the final dataset.</li>
  </ul>
  
### 3. Preprocessing and Modeling
- Preprocessing Steps:
   <ul>
   <li>Removing HTML tags</li>
   <li>Removing URLs</li>
   <li>Removing special characters</li>
   <li>Removing punctuation</li>
   <li>Removing numbers</li>
   <li>Removing stopwords</li>
   <li>Lemmatizing words</li>
   <li>Vectorizing the text data using TF-IDF Vectorizer</li>
   <li>Splitting the data into training and testing sets</li>
   <li>Scaling the data using StandardScaler</li>
   </ul>

- Modeling Steps:
   <ul>
   <li>Creating a baseline model</li>
   <li>Creating a Logistic Regression model</li>
   <li>Creating a K-Nearest Neighbors model</li>
   <li>Creating a Random Forest model</li>
   <li>Creating a Multinomial Naive Bayes model</li>
   <li>Creating a Gradient Boosting Classifier model</li>
   </ul>
 
 ### 4. Evaluation Understanding
 - Evaluating of the model's performance are based on the following metrics
    <ul>
    <li>Recall</li>
    <li>Precision</li>
    <li>Accuracy</li>
    <li>F1 Score</li>
    <li>Macro Average</li>
    <li>Weighted Average</li>
    </ul>
- Evaluated the performance of the models and chose the top-performing model using the mentioned metrics.
- Examined the model's coefficients to identify the key features for predicting the subreddit category.
- Studied the model's confusion matrix to identify prevalent misclassifications.
- Utilized the best-performing model to predict subreddit categories for test data and analyzed the outcomes.
- Analyzed the words with the highest and lowest coefficients from the best model to assess the sentiment and morale of the group.

### 5. Conclusion & Findings
- Best model: TD-IDF Vectorizer with Logistic Regression Model

- Logistic Regression: Achieved the highest score of 0.692201, indicating it performed the best among the models evaluated.

- Random Forest: Achieved a score of 0.687600, slightly lower than logistic regression but still showing strong performance.

- Gradient Boosting: Achieved a score of 0.677733, indicating good performance but slightly lower than logistic regression and random forest.

- Multinomial Naive Bayes: Achieved a score of 0.675761, showing competitive performance but lower than the previous models.

- KNN (K-Nearest Neighbors): Achieved the lowest score of 0.541597, indicating comparatively poorer performance compared to the other models evaluated in this context.

#### Sentiment Analysis findings
- In the "investing" subreddit, the model assigns a higher confidence score to negative sentiment (0.572924) compared to positive sentiment (0.427075). This suggests that, according to the model's analysis, there may be a stronger presence of negative sentiment or discussions about investment-related topics that lean towards pessimism or concerns within the "investing" community.

- In the "stocks" subreddit, the confidence score for positive sentiment (0.430530) is slightly higher than that for negative sentiment (0.569470). This indicates that, according to the model's analysis, there may be a slightly stronger presence of positive sentiment or discussions about stocks that lean towards optimism or positive outlooks within the "stocks" community.

#### Future work
Deploy these models within organizational contexts to monitor community sentiment in real-time. This can assist in detecting trends impacting investments in different stocks, facilitating better decision-making. Evaluate the evolution of model performance from its initial deployment to the present, showcasing its ability to adjust to shifts in language patterns within the community over time.
 
