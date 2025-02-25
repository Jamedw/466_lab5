import sys
from colaborative_filter import evaluation_csv, eval_report

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Methods: knn_user, knn_item, weighted_sum, mean_utility")
        sys.exit(1)
    elif len(sys.argv) != 4:
        print("Usage: python EvaluateCFList.py <Method> <Filename> <Repeats>")
        sys.exit(1)

    method = sys.argv[1]
    file = sys.argv[2]
    repeats = int(sys.argv[3])

    results = evaluation_csv(method, file, repeats)
    eval_report(results)