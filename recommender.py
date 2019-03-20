import pandas as pd 
import numpy as np
import warnings
import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

warnings.filterwarnings('ignore')

# userID --> ID of the user
# placeID --> ID of the restaurant
# rating --> average total rating given by the user to restaurant
# food_rating --> rating given by user to restaurant's food
# service_rating --> rating given by user based on restaurant's service
input_ratings = pd.read_csv('rating.csv')

#----------- Data Preprocessing ----------#
# Caclulate the number of unique userID and unique placeID
total_restaurants = input_ratings.placeID.unique()
total_users = input_ratings.userID.unique()

#---- Create a user-restaurant matrix to find the similarity between users annd restaurants for three types of ratings --- #
restuarant_id_sort = np.sort(total_restaurants)
user_id_sort = np.sort(total_users)

overall_rating = pd.DataFrame(np.zeros((len(total_restaurants), len(total_users))) - 1, columns=user_id_sort, index =restuarant_id_sort)
food_rating = pd.DataFrame(np.zeros((len(total_restaurants), len(total_users))) - 1, columns=user_id_sort, index =restuarant_id_sort)
service_rating = pd.DataFrame(np.zeros((len(total_restaurants), len(total_users))) - 1, columns=user_id_sort, index =restuarant_id_sort)

for pid, uid, o_rat, f_rat, s_rat in zip(input_ratings.placeID, input_ratings.userID, input_ratings.rating, input_ratings.food_rating, input_ratings.service_rating):
   overall_rating.loc[pid, uid] = o_rat
   food_rating.loc[pid, uid] = f_rat
   service_rating.loc[pid, uid] = s_rat

overall_rating = overall_rating.values 
food_rating = food_rating.values  
service_rating = service_rating.values

#create a matrix to check if the restaurant-user pair has a rating of zero or not
checkMatrix= np.zeros(overall_rating.shape)
checkMatrix[overall_rating >= 0] = 1

# ---- Train/Test Split ---- #
""" Split the existing ratings in a 70/30 train/test matrix. Also i make sure every restaruant and user receives a rating and gives a rating 
respectively in the training set"""

random.seed(0)
check = True
while check:
   training_set = checkMatrix.copy()

   for i in range(training_set.shape[1]):
      # use only rated restaurants
      index = list(np.where(training_set[:,i] == 1)[0]) 

      # randomly select 30% of whole dataset
      flag = int(round(len(index)*0.3))

      index_flag = random.sample(index,flag)
      training_set[index_flag,i] = 0  

   if np.sum(training_set,axis=1).min() > 1: 
      check = False

testing_set = checkMatrix - training_set
print(f"Training set: {training_set.sum()}\nTesting set: {testing_set.sum()}\n")

# ---- Use evaluation metrics to get the quantified values of model performances --- #
# Use root-mean-squared-error method for model performance employing linear regression
def root_mean_squared(y_true, y_predicte, R):
   rsme = np.sqrt(mean_squared_error(y_true[R == 1], y_predicte[R == 1]))
   return rsme

# --- Create a benchmark model which returns average predicted rating and use it to compare ratings ----- #

# Function for benchmark model mean 
def benchmark_mean (y_true, checkMatrix):
   y_true = y_true * checkMatrix
   return (np.sum(y_true, axis=1) / np.sum((checkMatrix == 1.0), axis=1)).reshape(y_true.shape[0], 1) * np.ones(y_true.shape)

# Function to get the benchmark root-mean-squared error to compare with the optimized results later
def get_benchmark_RSME(rating, setMatrix, setMatrix2):
   ratings_mean = benchmark_mean(rating, setMatrix)
   ratings_pred = np.zeros(rating.shape) + ratings_mean
   train = root_mean_squared(rating, ratings_pred, setMatrix)
   test = root_mean_squared(rating, ratings_pred, setMatrix2)
   print(f"RSME of training set is: {train}")
   print(f"RSME of testing set is: {test}\n\n")
   return ratings_pred

# Plot a boxplot to see the result of optimization when compared with benchmark model
def plot(y_predicte, y_true, setMatrix, title, filename):
   data1 = y_predicte[setMatrix == 1][y_true[setMatrix == 1] == 0]
   data2 = y_predicte[setMatrix == 1][y_true[setMatrix == 1] == 1]
   data3 = y_predicte[setMatrix == 1][y_true[setMatrix == 1] == 2]
   data = [data1,data2,data3]

   plt.boxplot(data)
   plt.xticks([1, 2, 3],[0,1,2])
   plt.xlabel('True Rating')
   plt.ylabel('Predicted Rating')
   plt.title(title)
   plt.savefig(filename, dpi=500)
   plt.clf()
   plt.cla()
   plt.close()

# ----- Benchmark Model performances for overall ratings ------ #
# Average ratings on training set and the prediction
print("-------BENCHMARK MODEL----------")
print("-----RSME rating------")
ratings_pred = get_benchmark_RSME(overall_rating, training_set, testing_set)
plot(ratings_pred, overall_rating, training_set, "Benchmark model for 'rating' with training set", "benchmark_images/bench_overall_training.png")
plot(ratings_pred, overall_rating, testing_set, "Benchmark model for 'rating' with testing set", "benchmark_images/bench_overall_testing.png")

# ------ Benchmark Model Performances for food rating ----- #
print("-----RSME food_rating------")
ratings_pred = get_benchmark_RSME(food_rating, training_set, testing_set)
plot(ratings_pred, food_rating, training_set, "Benchmark model for 'food_rating' with training set", "benchmark_images/bench_food_training.png")
plot(ratings_pred, food_rating, testing_set, "Benchmark model for 'food_rating' with testing set", "benchmark_images/bench_food_testing.png")

# ------ Benchmark Model Performances for service rating ----- #
print("-----RSME service_rating------")
ratings_pred = get_benchmark_RSME(service_rating, training_set, testing_set)
plot(ratings_pred, service_rating, training_set, "Benchmark model for 'service_rating' with training set", "benchmark_images/bench_service_training.png")
plot(ratings_pred, service_rating, testing_set, "Benchmark model for 'service_rating' with testing set", "benchmark_images/bench_service_testing.png")

# ----- Benchmark Model Performances for total average rating ---- #
print("-----RSME total average ratings-----")
total_ratings = (overall_rating + food_rating + service_rating) / 3
avg_ratings_pred = get_benchmark_RSME(total_ratings, training_set, testing_set)
plot(avg_ratings_pred, total_ratings, training_set, "Benchmark model for 'total_avg_ratings' with training set", "benchmark_images/bench_totalavg_training.png")
plot(avg_ratings_pred, total_ratings, testing_set, "Benchmark model for 'total_avg_ratings' with testing set", "benchmark_images/bench_totalavg_testing.png")
#print(total_ratings.shape)

# ------- We employ matrix factorization technqiue to calculate the optimized predicted ratings ------ #

# Function to get the cost function which is later minimized for optimization
def cost(parameters, true_values, setMatrix):
   num_restaurant = setMatrix.shape[0]
   num_user = setMatrix.shape[1]
   num_features = int((len(parameters) - num_restaurant) / (num_user + num_restaurant))

   # make a 2d matrix of parameters
   param_features = parameters[:num_restaurant * num_features].reshape(num_restaurant, num_features)
   
   # add bias term
   param_features = np.append(np.ones((num_restaurant, 1)), param_features, axis=1)
   
   #make a 2d matrix of user weights
   user_weights = parameters[num_restaurant * num_features:].reshape(num_user, num_features + 1)

   cost = 0.5 * np.sum((np.dot(param_features, user_weights.T) * setMatrix - true_values)**2)
   cost += (np.sum(user_weights[:,1:]**2) + np.sum(param_features[:,1:]**2))

   return cost

# Compute the gradient of the parameters
def Gradient (parameters, true_values, setMatrix):
   num_restaurant = int(setMatrix.shape[0])
   num_user = int(setMatrix.shape[1])
   num_features = int((len(parameters) - num_restaurant) / (num_user + num_restaurant))

   # make a 2d matrix of parameters and add a bias term
   param_features = parameters[:num_restaurant * num_features].reshape(num_restaurant, num_features)
   param_features = np.append(np.ones((num_restaurant, 1)), param_features, axis=1)
   
   #make a 2d matrix of user weights
   user_weights = parameters[num_restaurant * num_features:].reshape(num_user, num_features + 1)

   param_gradient = np.dot((np.dot(param_features, user_weights.T) * setMatrix - true_values), user_weights)
   weight_gradient = np.dot((np.dot(user_weights, param_features.T) * setMatrix.T - true_values.T), param_features)

   param_gradient += param_features
   param_gradient = param_gradient[:,1:]

   final_gradient = np.append(param_gradient.reshape(-1), weight_gradient.reshape(-1))

   return final_gradient

np.random.seed(99)
num_features = 3
#intialize the parameteres
param_features = np.random.normal(0,1, (len(total_restaurants), num_features))
user_weights = np.random.normal(0, 1, (len(total_users), num_features + 1))

# Make a 1 dimensional vector out of parameters
initial = np.append(param_features.reshape(-1), user_weights.reshape(-1))

# Function that results in the optimize RSME to see how well our algorithm for optimization works when compared to benchmark
def getRSME(rating, setMatrix, total_restaurants, num_features, total_users, initial, setMatrix2, cost, Gradient):
   mean = benchmark_mean(rating, setMatrix)
   optimize = minimize(cost, initial, jac=Gradient, args=(((rating * setMatrix) - mean) * setMatrix, setMatrix))
   param_features_opt = optimize.x[:len(total_restaurants) * num_features].reshape(len(total_restaurants), num_features)
   param_features_opt = np.append(np.ones((len(total_restaurants),1)),param_features_opt,axis=1)
   user_weights_opt = optimize.x[len(total_restaurants) * num_features:].reshape(len(total_users), num_features + 1)
   ratings_pred = np.dot(param_features_opt, user_weights_opt.T) + mean
   #print(ratings_pred.shape)
   #print(ratings_pred)
   train = root_mean_squared(rating, ratings_pred, setMatrix)
   test = root_mean_squared(rating, ratings_pred, setMatrix2)
   print(f"RSME of training set is: {train}")
   print(f"RSME of testing set is: {test}\n")
   return ratings_pred

# Function that generates a dictionary of top 10 recommended restaurants with key being the restaurant id and value being the predicted rating
def top_recommend(ratings_pred, restaurant_id):
   ratings_pred = ratings_pred.tolist()
   for i in range(len(ratings_pred)):
      ratings_pred[i] = np.mean(ratings_pred[i])
   
   # Create a dictionary of restaurant ids and predicted ratings
   dict_predicte = dict(zip(restaurant_id, ratings_pred))

   #Create a dictionary of top 10 recommended restuarants
   recommended = {}
   for i in range(10):
      key_max = max(dict_predicte.keys(), key=(lambda k: dict_predicte[k]))
      recommended[key_max] = dict_predicte[key_max]
      del dict_predicte[key_max]
   
   return recommended

# Working for RSME of overall ratings
print("---------OPTIMIZED MODEL AFTER MATRIX FACTORIZATION------------")
print("----RMSE optimized rating------")
ratings_pred = getRSME(overall_rating, training_set, total_restaurants, num_features, total_users, initial, testing_set, cost, Gradient)
plot(ratings_pred, overall_rating, training_set, "Optimized model for 'rating' with training set", "optimized_images/opt_overall_training.png")
plot(ratings_pred, overall_rating, testing_set, "Optimized model for 'rating' with testing set", "optimized_images/opt_overall_testing.png")
recommended = top_recommend(ratings_pred, total_restaurants)
print("-----Top 10 recommended restaurant based on overall rating-----")
print(recommended)
print("\n\n")

# Working for RSME of food rating
print("----RMSE optimized food rating------")
ratings_pred = getRSME(food_rating, training_set, total_restaurants, num_features, total_users, initial, testing_set, cost, Gradient)
plot(ratings_pred, overall_rating, training_set, "Optimized model for 'food_rating' with training set", "optimized_images/opt_food_training.png")
plot(ratings_pred, overall_rating, testing_set, "Optimized model for 'food_rating' with testing set", "optimized_images/opt_food_testing.png")
recommended = top_recommend(ratings_pred, total_restaurants)
print("-----Top 10 recommended restaurant based on food rating-----")
print(recommended)
print("\n\n")

# Working for RSME of service rating
print("----RMSE optimized service rating------")
ratings_pred = getRSME(service_rating, training_set, total_restaurants, num_features, total_users, initial, testing_set, cost, Gradient)
plot(ratings_pred, overall_rating, training_set, "Optimized model for 'service_rating' with training set", "optimized_images/opt_service_training.png")
plot(ratings_pred, overall_rating, testing_set, "Optimized model for 'service_rating' with testing set", "optimized_images/opt_service_testing.png")
recommended = top_recommend(ratings_pred, total_restaurants)
print("-----Top 10 recommended restaurant based on service rating-----")
print(recommended)
print("\n\n")

# Working for RSME of total average rating
print("----RMSE optimized total average rating------")
ratings_pred = getRSME(total_ratings, training_set, total_restaurants, num_features, total_users, initial, testing_set, cost, Gradient)
plot(ratings_pred, overall_rating, training_set, "Optimized model for 'total_avg_rating' with training set", "optimized_images/opt_totalavg_training.png")
plot(ratings_pred, overall_rating, testing_set, "Optimized model for 'total_avg_rating' with testing set", "optimized_images/opt_totalavg_testing.png")
recommended = top_recommend(ratings_pred, total_restaurants)
print("-----Top 10 recommended restaurant based on total average rating-----")
print(recommended)
print("\n\n")

   









