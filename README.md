Project Overview

Online grocery platforms contain millions of purchase transactions that can be used to understand customer purchasing behavior and recommend relevant products.

This project processes 32M+ grocery purchase records using PostgreSQL and Python and implements multiple recommendation approaches:

Popularity-based recommendations
Reorder-based recommendations
Collaborative filtering
Reorder prediction using Logistic Regression

The system uses PostgreSQL for large-scale data aggregation and filtering, while Python and Scikit-learn are used for recommendation algorithms and machine learning.


Technologies Used :-

Programming & Data
Python
Pandas
NumPy
SciPy
Database
PostgreSQL
SQL
SQLAlchemy
Psycopg2
Machine Learning
Scikit-learn
Logistic Regression
Nearest Neighbors
Cosine similarity
Development
Git
GitHub


The project uses the Instacart Market Basket Analysis dataset.

The dataset contains grocery purchase information including:

Customers
Orders
Products
Product reorders
Departments
Aisles

The project processes more than 32 million order-product records.

PostgreSQL Database

The raw CSV data is imported into PostgreSQL to allow efficient querying and aggregation.

Main tables:

orders
products
order_products

Additional dataset information includes:

departments
aisles
Example SQL aggregation


Recommendation Methods
1. Popularity-Based Recommendations

Products are ranked according to their total number of purchases.

Example results:

Product	Purchases
Banana	472,565
Bag of Organic Bananas	379,450
Organic Strawberries	264,683
Organic Baby Spinach	241,921
Organic Hass Avocado	213,584


2. Reorder-Based Recommendations

Products are also ranked according to the number of times customers reordered them.

Example:

Product	Reorders
Banana	398,609
Bag of Organic Bananas	315,913
Organic Strawberries	205,845
Organic Baby Spinach	186,884
Organic Hass Avocado	170,131

This provides an alternative recommendation strategy based on repeat purchasing behavior.

3. Collaborative Filtering

Collaborative filtering uses customer purchase behavior to identify products that are frequently purchased by similar users.

The resulting dataset contains:

206,209 users
49,677 products
13,307,953 non-zero user-product interactions

A sparse matrix is then used in Python to reduce memory usage.

NearestNeighbors with cosine distance is used to identify similar products.


Reorder Prediction

A Logistic Regression model is used to predict whether a product has a high probability of being reordered.

Product-level statistics are calculated in PostgreSQL.


Installation:

Clone the repository:

git clone https://github.com/saudaminighosh/Recommendation_System.git
cd Recommendation_System

Install the required Python packages:

pip install -r requirements.txt

Required packages include:

pandas
numpy
scipy
scikit-learn
sqlalchemy
psycopg2-binary
matplotlib
jupyter


Running the Project

From the project root:

python -m src.main

The system will execute:

1. Popularity-based recommendations
2. Reorder-based recommendations
3. Collaborative filtering
4. Reorder prediction model

Author : Saudamini Ghosh (https://github.com/saudaminighosh)