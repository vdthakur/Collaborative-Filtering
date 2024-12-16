# import time
import csv
import sys
from math import sqrt
from collections import defaultdict
from pyspark import SparkContext

# Initialize Spark context
collab_filter = SparkContext('local[*]', 'collaborative_filtering')

# Read command line arguments: training file, validation file, output file
input_file = sys.argv[1]    # "yelp_train.csv" 
validation_file = sys.argv[2] # "yelp_val.csv" 
output = sys.argv[3]        # "cf_predictions.csv" 

# start = time.time()

# Load training data into an RDD
yelp_rdd = collab_filter.textFile(input_file)
first_line = yelp_rdd.first()
# Filter out header and split lines by comma
yelp_rdd = yelp_rdd.filter(lambda line: line != first_line).map(lambda row: row.strip().split(','))

# For each user, prepare tuples of (user_id, (rating, count=1))
user_ratings = yelp_rdd.map(lambda row: (row[0], (float(row[2]), 1)))

# Aggregate sum of ratings and count of ratings per user
user_rating_totals = user_ratings.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))

# Compute each user's mean rating
user_mean_ratings = user_rating_totals.mapValues(lambda x: x[0] / x[1]).collectAsMap()

# Build a dictionary: user -> {business: (rating - user_mean)}
# This normalizes each user's ratings by subtracting their mean rating
user_to_business_ratings = (yelp_rdd.map(lambda row: (row[0], (row[1], float(row[2]) - user_mean_ratings[row[0]]))).groupByKey()
             .mapValues(dict).collectAsMap())

# Build a dictionary: business -> {user: (rating - user_mean)}
business_to_user_ratings = (yelp_rdd.map(lambda row: (row[1], (row[0], float(row[2]) - user_mean_ratings[row[0]]))).groupByKey()
             .mapValues(dict).collectAsMap())

# Store actual ratings in a dictionary keyed by (user, business)
user_business_pair_ratings = (yelp_rdd.map(lambda row: ((row[0], row[1]), float(row[2]))).collectAsMap())

# Compute the average rating for each business
business_mean_ratings = (
    yelp_rdd.map(lambda row: (row[1], (float(row[2]), 1))).reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
            .mapValues(lambda x: x[0] / x[1]).collectAsMap())

# Compute the global average rating (over all users and businesses in the training data)
global_average = yelp_rdd.map(lambda row: float(row[2])).mean()

# Load validation data
validation_rdd = collab_filter.textFile(validation_file)
first_line = validation_rdd.first()
validation_rdd = (validation_rdd.filter(lambda line: line != first_line).map(lambda row: row.strip().split(',')).map(lambda row: (row[0], row[1])))

def pearson_correlation(b1, b2, business_to_user_ratings):
    """
    Compute the Pearson correlation coefficient between two businesses b1 and b2.
    The ratings are taken from business_to_user_ratings, which maps each business
    to a dict of {user: normalized_rating}.
    """
    ratings1 = business_to_user_ratings.get(b1, {})
    ratings2 = business_to_user_ratings.get(b2, {})

    # Find common users who rated both businesses
    common_users = set(ratings1.keys()).intersection(ratings2.keys())
    n = len(common_users)
    if n == 0:
        return None

    # Compute sums and sums of squares for the common ratings
    sum1 = sum(ratings1[user] for user in common_users)
    sum2 = sum(ratings2[user] for user in common_users)
    sum1_squared = sum(ratings1[user] ** 2 for user in common_users)
    sum2_squared = sum(ratings2[user] ** 2 for user in common_users)
    c_sum = sum(ratings1[user] * ratings2[user] for user in common_users)

    # Pearson numerator and denominator components
    num = c_sum - (sum1 * sum2 / n)
    den_p1 = sum1_squared - (sum1 ** 2) / n
    den_p2 = sum2_squared - (sum2 ** 2) / n
    denominator = den_p1 * den_p2

    # Check for valid denominator
    if denominator <= 0:
        return None

    den = sqrt(denominator)
    if den == 0:
        return None

    # Pearson similarity
    similarity = num / den
    return similarity

def make_predictions(user_business_pair, user_to_business_ratings, business_to_user_ratings, user_mean_ratings, business_mean_ratings, global_average, num_neighbors):
    """
    Predict the rating for a given (user, business) pair using a user-based CF approach.
    Uses Pearson correlation to compute similarities between businesses rated by the user.
    Applies various fallback strategies for cold-start scenarios.
    """
    user, business = user_business_pair

    # Fallback to global average if user or business not found
    user_avg = user_mean_ratings.get(user, global_average)
    business_avg = business_mean_ratings.get(business, global_average)

    # Cold start cases
    if user not in user_to_business_ratings and business not in business_to_user_ratings:
        # Both user and business never seen
        return global_average
    elif user not in user_to_business_ratings:
        # New user, known business
        return business_avg
    elif business not in business_to_user_ratings:
        # Known user, new business
        return user_avg

    # Gather the user's normalized ratings for the businesses they've rated
    user_ratings = user_to_business_ratings[user]
    similarities = []

    # Compute similarity between the target business and each business the user has rated
    for rated_business in user_ratings.keys():
        if rated_business == business:
            continue
        similarity = pearson_correlation(business, rated_business, business_to_user_ratings)
        
        # If we cannot compute similarity, fallback to ratio-based heuristic
        if similarity is None:
            ratio = business_avg / business_mean_ratings.get(rated_business, global_average)
            if 0.9 <= ratio <= 1.1:
                similarity = 0.9
            elif 0.8 <= ratio < 0.9 or 1.1 < ratio <= 1.2:
                similarity = 0.7
            elif 0.7 <= ratio < 0.8 or 1.2 < ratio <= 1.3:
                similarity = 0.5
            elif 0.6 <= ratio < 0.7 or 1.3 < ratio <= 1.4:
                similarity = 0.3
            else:
                similarity = 0.1

        # Assign lower weight to negative similarities
        weight = 1.0 if similarity >= 0 else 0.5
        similarities.append((similarity * weight, user_ratings[rated_business]))

    # If no similarities found, fallback to a weighted average of user and business averages
    if not similarities:
        fallback = (0.6 * user_avg) + (0.4 * business_avg)
        return fallback

    # Select top-N neighbors based on absolute similarity
    top_neighbors = sorted(similarities, key=lambda x: -abs(x[0]))[:num_neighbors]
    selected_neighbors = top_neighbors if top_neighbors else similarities

    # Compute the weighted average of the selected neighbors' ratings
    numerator = sum(sim * rating for sim, rating in selected_neighbors)
    denominator = sum(abs(sim) for sim, _ in selected_neighbors)
    if denominator == 0:
        return user_avg

    # Compute predicted rating, ensure it's within [1, 5]
    predicted_rating = user_avg + (numerator / denominator)
    predicted_rating = max(1.0, min(5.0, predicted_rating))
    return predicted_rating

def generate_cf_predictions(user_business_pair):
    """
    Wrapper function that uses make_predictions() to get the predicted rating
    for each (user, business) pair in the validation data.
    """
    user_id, business_id = user_business_pair
    predicted_rating = make_predictions(
        user_business_pair,
        user_to_business_ratings,
        business_to_user_ratings,
        user_mean_ratings,
        business_mean_ratings,
        global_average,
        num_neighbors=18)
    return ((user_id, business_id), predicted_rating)

# Apply the prediction function to all validation pairs
cf_predictions_rdd = validation_rdd.map(generate_cf_predictions).collect()

# Write predictions to output CSV
with open(output, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['user_id', 'business_id', 'prediction'])
    for (user_id, business_id), predicted_rating in cf_predictions_rdd:
        writer.writerow([user_id, business_id, predicted_rating])

# end = time.time()
# print(f"Execution time: {end - start} seconds")