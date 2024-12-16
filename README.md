# Collaborative Filtering with PySpark

This repository contains a Python script that implements a collaborative filtering (CF) approach to rating prediction using Apache Spark. It leverages PySpark RDD transformations to preprocess user-business rating data, compute mean ratings, calculate similarity measures, and ultimately predict unknown ratings for a given set of (user, business) pairs. The example provided uses Yelp data, but the approach is generalizable to any user-item rating dataset.

## Overview

**Key Features:**
- Uses PySpark for parallel data processing and transformations.
- Implements a user-based collaborative filtering method.
- Normalizes ratings by subtracting each user's average rating to handle user-level rating scale differences.
- Employs Pearson correlation to measure similarity between items (businesses).
- Provides various fallback strategies for "cold-start" problems when encountering new users or items.
- Outputs the final predictions to a CSV file.

**Files:**
- `yelp_train.csv`: Training dataset containing existing user-business ratings.
- `yelp_val.csv`: Validation dataset for which the script generates predictions.
- `cf_predictions.csv`: Output file containing `(user_id, business_id, predicted_rating)`.

## Data Format

**Input File (Training/Validation):**
- CSV format with a header line.
- Columns: `user_id, business_id, stars` (where `stars` is the user’s rating of the business).

**Output File:**
- CSV file with the following header: `user_id, business_id, prediction`
- Each line: the predicted rating for the `(user_id, business_id)` pair.

## How It Works

1. **Data Loading & Preprocessing:**  
   The script starts a local Spark context and reads the training data (`yelp_train.csv`), filtering out the header.  
   
   It then computes:
   - **User Mean Ratings:** For each user, sum all ratings and count the number of items rated. The mean rating is used for normalization.
   - **Business Mean Ratings:** Similar computations as for users, used as a fallback and for similarity ratio heuristics.
   - **Global Average Rating:** A global fallback rating if both user and business are unseen in the training data.

2. **Data Structures Built:**
   - `user_mean_ratings`: A dictionary mapping `user_id` to their average rating.
   - `user_to_business_ratings`: A dictionary mapping `user_id` to another dictionary of `(business_id: normalized_rating)` pairs.
   - `business_to_user_ratings`: A dictionary mapping `business_id` to `(user_id: normalized_rating)` pairs.
   
   **Normalization:** Each rating is transformed into `(original_rating - user_mean_rating)` so that ratings reflect how much a user deviates from their own norm.

3. **Pearson Correlation Similarity:**
   To predict a user’s rating for a target business, the script:
   - Identifies other businesses the user has rated.
   - Computes the Pearson correlation between the target business and each of these "neighbor" businesses.  
   
   If insufficient data exists to compute Pearson correlation, it uses heuristic-based fallback similarities.

4. **Making Predictions:**
   The predicted rating is computed as:
   predicted_rating = user_mean + (Σ(similarity * normalized_rating) / Σ|similarity|)

   If no neighbors or similarities can be found, the script falls back to combinations of user average, business average, or global average.

5. **Output:**
For each `(user, business)` pair in the validation dataset, the script predicts a rating and writes it to `task2_1.csv`.

## Usage

**Prerequisites:**
- Python 3.x
- PySpark
- A Spark environment or local Spark session
- CSV data files in the format described

**Running the Script:**
```bash
spark-submit task2.py yelp_train.csv yelp_val.csv cf_predictions.csv

Here:  
- **yelp_train.csv** is the training data.  
- **yelp_val.csv** is the validation data.  
- **cf_predictions.csv** is the output file containing the predictions.

**Script Arguments:**  
1. `sys.argv[1]`: Input training file (e.g., `yelp_train.csv`)  
2. `sys.argv[2]`: Validation file (e.g., `yelp_val.csv`)  
3. `sys.argv[3]`: Output predictions file (e.g., `task2_1.csv`)

**Customization and Tuning**

- **Number of Neighbors (`num_neighbors`):**  
  The code uses `num_neighbors=18` by default. You can experiment with different values to see how it affects prediction accuracy.

- **Similarity Thresholds & Heuristics:**  
  The script applies ratio-based heuristic similarities when Pearson correlation can’t be computed. You can adjust these ratios and similarity fallback logic based on your dataset or performance needs.

- **Fallback Strategies:**  
  Adjust the weights for user average, business
