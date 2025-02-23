"""
EXAMPLE USAGE

model = collaborative_filtering("knn_item")
model.fit(collaborative_filtering.parse_data("./jester-data-1.csv"))
model.predict(0,99)    

"""



class collaborative_filtering:
    
    def __init__(self, method="None"):
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
    
    def get_N_neighbors(self, id_num, K=10 ,similarity_type="user"):
        similarities = []
        if similarity_type == "user":
            for i in range(len(self.x_train)):
                if i == id_num:
                    continue
                else:
                    similarities.append((self.user_similarity(id_num, i), i))
            sorted_neighbors = sorted(similarities, reverse=True)
            return sorted_neighbors[:K]
        else:
            for i in range(len(self.x_train[0])):
                if i == id_num:
                    continue
                else:
                    similarities.append((self.item_similarity(id_num, i), i))
            sorted_neighbors = sorted(similarities, reverse=True)
            return sorted_neighbors[:K]
        
    def knn_user(self, user_id, item_id):
        ## assume best K is 10
        nearest_sims = self.get_N_neighbors(user_id)
        users = np.ones((len(nearest_sims), self.x_train.shape[1]))
        sims = np.ones(len(nearest_sims))

        for i in range(len(nearest_sims)):
            sims[i] = nearest_sims[i][0]
            users[i] = self.x_train[nearest_sims[i][1]]
            
        return np.nanmean(self.x_train[user_id]) + (1/abs(sims).sum()) * np.nansum((sims * (users[:, item_id] - np.nanmean(users, axis = 1))))

    def knn_item(self, user_id, item_id):
        ## assume best K is 10
        nearest_sims = self.get_N_neighbors(item_id, similarity_type="item")
        items = np.ones((len(nearest_sims), self.x_train.shape[0]))
        sims = np.ones(len(nearest_sims))

        for i in range(len(nearest_sims)):
            sims[i] = nearest_sims[i][0]
            items[i] = self.x_train[:,nearest_sims[i][1]]

        return np.nanmean(self.x_train[:, item_id]) + (1/abs(sims).sum()) * np.nansum((sims * (items[:, user_id] - np.nanmean(items, axis = 1))))
        
        
    
    def predict(self, user_id, item_id):
        
        if self.method == "knn_user":
            return self.knn_user(user_id, item_id)
            pass
        elif self.method == "knn_item":
            return self.knn_item(user_id, item_id)
            pass
        elif self.method == "method 3":
            pass
        else:
            pass
        
        
        