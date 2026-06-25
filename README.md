# Content Engagement Optimization Engine

This project contains a data analysis pipeline and two neural network models to analyze and predict social media metrics.

## Project Structure

### Phase 1: Engagement Regression
* Objective: Predict the exact engagement rate of a post.
* Architecture: 3-layer sequential neural network using mean squared error as the loss function.
* Core Finding: Using downstream performance metrics like likes, shares, and comments causes data leakage if the goal is to evaluate content before it is published.

### Phase 2: Virality Classification
* Objective: Predict if a post will be in the top 10% of performance before it goes live.
* Feature Engineering: Dropped the metrics that cause data leakage (likes, shares, comments, impressions).
* Class Imbalance Fix: Handled the skewed 90/10 data distribution by applying class weights to ensure the minority class is heavily factored into the loss calculation.
* Checkpoints: Implemented a model checkpoint callback to track validation accuracy across training rounds and automatically save the highest-performing weight matrix.

## Setup
Install dependencies using:
pip install pandas numpy scikit-learn tensorflow matplotlib
