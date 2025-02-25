import sys
from colaborative_filter import evaluation, eval_report

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Methods: knn_user, knn_item, weighted_sum, mean_utility")
        sys.exit(1)
    elif len(sys.argv) != 4:
        print("Usage: python EvaluateCFRandom.py <Method> <Size> <Repeats>")
        sys.exit(1)
    
    method = sys.argv[1]
    size = int(sys.argv[2])
    repeats = int(sys.argv[3])

    results = evaluation(method, size, repeats)
    eval_report(results)