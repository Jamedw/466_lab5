import sys
from colaborative_filter import evaluation
if __name__ == "__main__":
    if len(sys.argv) == 2:
        print("Methods: knn_user, knn_item, weighted_sum, mean_utility")
        sys.exit(1)
    elif len(sys.argv) != 4:
        print("Usage: python EvaluateCFRandom.py <Method> <Size> <Repeats>")
        sys.exit(1)
    
    method = sys.argv[1]
    size = sys.argv[2]
    repeats = sys.argv[3]

    evaluation(method, size, repeats)