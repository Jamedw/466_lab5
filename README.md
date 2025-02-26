# Team Members
James Dwyer: jadwyer@calpoly.edu
Jerry Huo: jehuo@calpoly.edu


"in order to find what metrics are avaliable run program with random [method] arg an it will raise an exception which contains a list of all methods"
methods = {"knn_item","knn_user", "weighted_sum", "mean_utility"}


Executables:
EvaluateCFLIST.py 
args: Method Filename
Method: "a method which the collaborative filtering algorithm will use to estimate 
u(c,j)"
filename: "path to file containing comma seperated list of (user_id,item_id)"
file dependencies: colaborative_filter.py
python libraries: sys, colaborative_filter


EvaluateCFRandome.py 
args: Method Size Repeats
Method: "a method which the collaborative filtering algorithm will use to estimate 
u(c,j)"
size: "the amount of u(c,j) to calculate"
repeats "the ammount of times size should occur"
file dependencies: colaborative_filter.py
python libraries: sys, colaborative_filter


colaborative_filter.py
description: "this file contains all the code require to make a collaborative filtering mode. the class colaborative_filter implements ways to evaluate a model given a metric"
file dependencies: jester-data-1.csv colaborative_filter.py
python libraries: numpy, pandas, random 

