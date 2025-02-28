"""
EXAMPLE USAGE

model = collaborative_filtering("knn_item")
model.fit(collaborative_filtering.parse_data("./jester-data-1.csv"))
model.predict(0,99)    

"""

import numpy as np
import pandas as pd
import random

class collaborative_filtering:
    
    methods = {"knn_item","knn_user", "weighted_sum", "mean_utility"}
    
    def __check_method(self, method):
        if method not in collaborative_filtering.methods:
            raise Exception(f"Valid Methods: {collaborative_filtering.methods}")
    
    def __init__(self, method="None"):
        self.__check_method(method)
        self.method = method
        
    ## takes in as input a filepath 
    ## outputs correct format for collaborative filtering model
    def parse_data(file_path):
        dataframe = pd.read_csv(file_path,header=None)
        dataframe = dataframe.drop(dataframe.columns[0], axis=1)
        values = dataframe.values
        return np.where(values == 99, np.nan, values)
        
    ##x_train should be the data structure from parse data
    def fit(self, x_train):
    ## save a copy so that we know the model data wont be modified
        self.x_train = x_train.copy()
    
    def user_similarity(self, user_id_1, user_id_2):
        ## get the rows representing user1 and user2
        user_1 = self.x_train[user_id_1]
        user_2 = self.x_train[user_id_2]
        shared_jokes = (~np.isnan(user_1)) & (~np.isnan(user_2))
        
        
        user_1_shared = user_1[shared_jokes]
        user_2_shared = user_2[shared_jokes]
        user_1_mean = np.nanmean(user_1)
        user_2_mean = np.nanmean(user_2)
        
        
        numerator = ((user_1_shared - user_1_mean) * (user_2_shared - user_2_mean)).sum()
        denominator = np.sqrt(np.sum(np.square(user_1_shared - user_1_mean))) * np.sqrt(np.sum(np.square(user_2_shared - user_2_mean)))
        return numerator / denominator
    
    
    def item_similarity(self, item_id_1, item_id_2):
        ## get the rows representing user1 and user2
        item_1 = self.x_train[:, item_id_1]
        item_2 = self.x_train[:, item_id_2]
        shared_ratings = (~np.isnan(item_1)) & (~np.isnan(item_2))
        
        
        item_1_shared = item_1[shared_ratings]
        item_2_shared = item_2[shared_ratings]
        item_1_mean = np.nanmean(item_1)
        item_2_mean = np.nanmean(item_2)
        
        
        numerator = ((item_1_shared - item_1_mean) * (item_2_shared - item_2_mean)).sum()
        denominator = np.sqrt(np.sum(np.square(item_1_shared - item_1_mean))) * np.sqrt(np.sum(np.square(item_2_shared - item_2_mean)))
        return numerator / denominator
    
    def get_N_neighbors(self, user_id, item_id , K=50 ,similarity_type="user"):
        similarities = []
        if similarity_type == "user":
            for i in range(len(self.x_train)):
                if i == user_id:
                    continue
                else:
                    if ~np.isnan(self.x_train[i][item_id]):
                        similarities.append((self.user_similarity(user_id, i), i))
            sorted_neighbors = sorted(similarities, reverse=True)
            return sorted_neighbors[:K]
        else:
            for i in range(len(self.x_train[0])):
                if i == item_id:
                    continue
                else:
                    if ~np.isnan(self.x_train[user_id][i]):
                        similarities.append((self.item_similarity(item_id, i), i))
            sorted_neighbors = sorted(similarities, reverse=True)
            return sorted_neighbors[:K]
        
    def knn_user(self, user_id, item_id):
        ## assume best K is 10
        nearest_sims = self.get_N_neighbors(user_id, item_id)
        users = np.ones((len(nearest_sims), self.x_train.shape[1]))
        sims = np.ones(len(nearest_sims))

        for i in range(len(nearest_sims)):
            sims[i] = nearest_sims[i][0]
            users[i] = self.x_train[nearest_sims[i][1]]
            
        return np.nanmean(self.x_train[user_id]) + (1/abs(sims).sum()) * np.nansum((sims * (users[:, item_id] - np.nanmean(users, axis = 1))))

    def knn_item(self, user_id, item_id):
        ## assume best K is 10
        nearest_sims = self.get_N_neighbors(user_id, item_id, similarity_type="item")
        items = np.ones((len(nearest_sims), self.x_train.shape[0]))
        sims = np.ones(len(nearest_sims))

        for i in range(len(nearest_sims)):
            sims[i] = nearest_sims[i][0]
            items[i] = self.x_train[:,nearest_sims[i][1]]

        return np.nanmean(self.x_train[:, item_id]) + (1/abs(sims).sum()) * np.nansum((sims * (items[:, user_id] - np.nanmean(items, axis = 1))))
    
    # adjusted weighted sum
    def weighted_sum(self, user_id, item_id):
        user_ratings = self.x_train[user_id]
        user_rated = user_ratings[~np.isnan(user_ratings)]
        user_mean = np.mean(user_rated) if len(user_rated) > 0 else 0
        
        sum_weighted_ratings = 0.0
        sum_abs_similarities = 0.0

        for other_user_id in range(len(self.x_train)):
            if other_user_id == user_id:
                continue
            if np.isnan(self.x_train[other_user_id, item_id]):
                continue
            similarity = self.user_similarity(user_id, other_user_id)
            
            # no shared items
            if np.isnan(similarity):
                continue

            other_user_ratings = self.x_train[other_user_id]
            other_rated = other_user_ratings[~np.isnan(other_user_ratings)]
            other_user_mean = np.mean(other_rated) if len(other_rated) > 0 else 0
            other_rating = self.x_train[other_user_id, item_id]

            sum_weighted_ratings += similarity * (other_rating - other_user_mean)
            sum_abs_similarities += abs(similarity)

        k = 1.0 / sum_abs_similarities
        prediction = user_mean + k * sum_weighted_ratings
        return prediction

    def mean_utility(self, item_id):
        item_ratings = self.x_train[:, item_id]
        rated = item_ratings[~np.isnan(item_ratings)]
        if len(rated) == 0:
            return 0
        return np.mean(rated)
    
    def predict(self, user_id, item_id):
        
        ## per specifications, if matrix at user_id, item_id is
        ## not empty replace it with nan then preform collaborative filtering 
        
        index_is_nan = True
        if self.x_train[user_id][item_id] != np.nan and self.x_train[user_id][item_id] != 99:
                index_is_nan = False
                prev_value = self.x_train[user_id][item_id]
                self.x_train[user_id][item_id] = np.nan
        
        if self.method == "knn_user":
            prediction = self.knn_user(user_id, item_id)
        elif self.method == "knn_item":
            prediction = self.knn_item(user_id, item_id)
        elif self.method == "weighted_sum":
            prediction = self.weighted_sum(user_id, item_id)
        else:
            prediction = self.mean_utility(item_id)

        if not index_is_nan:
            self.x_train[user_id][item_id] = prev_value
        
        return prediction

def evaluation(method, size, repeats):
    model = collaborative_filtering(method)
    model.fit(collaborative_filtering.parse_data("./jester-data-1.csv"))
    rows = model.x_train.shape[0]
    columns = model.x_train.shape[1]
    results = []
    runs_deltas = np.ones((repeats,size))
    for j in range(repeats):
        run_delta = np.ones(size)
        ## using a for loop rn hhowever np.choice may work with a masked 
        ## x_train
        for i in range(size):
            invalid_pair = True
            while invalid_pair:
                row = random.randint(0, rows-1)
                column = random.randint(0, columns-1)
                if ~np.isnan(model.x_train[row][column]):
                    user_id = row
                    item_id = column 
                    actual_rating = model.x_train[user_id][item_id]
                    predicted_rating = model.predict(user_id, item_id)
                    delta_rating = actual_rating - predicted_rating
                    run_delta[i] = delta_rating
                    print(f"User: {user_id}, Item: {item_id}, Actual: {actual_rating}, Predicted: {predicted_rating}, Delta: {delta_rating}")
                    invalid_pair = False
                    result = {
                        'user_id': user_id,
                        'item_id': item_id,
                        'actual_rating': actual_rating,
                        'predicted_rating': predicted_rating,
                        'delta_rating': delta_rating
                    }
                    results.append(result)
        print(f"MAE: {np.nanmean(abs(run_delta))}")
        print(f"Standard Deviation; {np.sqrt(np.nansum(np.square(run_delta - np.nanmean(run_delta))) / len(run_delta))}")
        runs_deltas[j] = run_delta
    return results
    
    """
    The above code is to specification for the first eval description 
    
    flat_runs_deltas = runs_deltas.flatten()
    print(f"Overall MAE: {abs(flat_runs_deltas).mean()}")
    """
    
def evaluation_csv(method, filepath, repeats=1):
    model = collaborative_filtering(method)
    model.fit(collaborative_filtering.parse_data("./jester-data-1.csv"))

    points = pd.read_csv(filepath, header=None)
    results = []
    
    for _ in range(repeats):
        run_delta = np.ones(len(points))
        for index, row in points.iterrows():
            user_id = row[0]
            item_id = row[1] 
            actual_rating = model.x_train[user_id][item_id]
            predicted_rating = model.predict(user_id, item_id)
            delta_rating = actual_rating - predicted_rating
            run_delta[index] = delta_rating
            print(f"User: {user_id}, Item: {item_id}, Actual: {actual_rating}, Predicted: {predicted_rating}, Delta: {delta_rating}")
            result = {
                'user_id': user_id,
                'item_id': item_id,
                'actual_rating': actual_rating,
                'predicted_rating': predicted_rating,
                'delta_rating': delta_rating
            }
            results.append(result)
    
        print(f"MAE: {np.nanmean(abs(run_delta))}")
        print(f"Standard Deviation; {np.sqrt(np.nansum(np.square(run_delta - np.nanmean(run_delta))) / len(run_delta))}")
    return results

def eval_report(results):
    total = len(results)
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    for result in results:
        actual = result['actual_rating'] >= 5.0
        predicted = result['predicted_rating'] >= 5.0
        
        if actual and predicted:
            true_positives += 1
        elif not actual and predicted:
            false_positives += 1
        elif not actual and not predicted:
            true_negatives += 1
        elif actual and not predicted:
            false_negatives += 1

    print("\nConfusion Matrix:")
    print(f"True Positives: {true_positives}")
    print(f"False Positives: {false_positives}")
    print(f"True Negatives: {true_negatives}")
    print(f"False Negatives: {false_negatives}")
    

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * (precision * recall) / (precision + recall)
    accuracy = (true_positives + true_negatives) / total

    print("\nRecommendation Metrics:")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Overall Accuracy: {accuracy}")