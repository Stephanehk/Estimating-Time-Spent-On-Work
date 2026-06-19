import argparse
import json
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
import itertools

# This script lives in validation/; add the repo root so `utils` is importable and
# the relative data/ paths resolve when run from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.constants import MIN_TIME_PER_TASK, HOURS_PER_DAY


def sample_directions(n, K, kind="gaussian", rng=None):
    rng = np.random.default_rng(rng)
    ds = []
    for _ in range(K):
        if kind == "gaussian":
            d = rng.normal(size=n)
        elif kind == "rademacher":
            d = rng.choice([-1.0, 1.0], size=n)
        else:
            raise ValueError(f"Unknown kind={kind}")
        # avoid near-zero vector; normalize
        norm = np.linalg.norm(d)
        if norm < 1e-12:
            continue
        ds.append(d / norm)
    return ds


def get_max_weight_solution(n, A_eq, b_eq, A_ub, b_ub, bounds, LP_json):
    
    max_weight_solutions = []
    for i in range(n):
        c_i = np.zeros(len(LP_json["c"])-1)
        c_i[i] = -1  # maximize weight i
        res_i = linprog(
            c_i,
            A_eq=A_eq, b_eq=b_eq,
            A_ub=A_ub, b_ub=b_ub,
            bounds=bounds,
        )
        assert res_i.success, f"Failed to solve LP for weight {i}"
        max_weight_solutions.append(res_i.x)

        res_i = linprog(
            -c_i,
            A_eq=A_eq, b_eq=b_eq,
            A_ub=A_ub, b_ub=b_ub,
            bounds=bounds,
        )
        assert res_i.success, f"Failed to solve LP for weight {i}"
        max_weight_solutions.append(res_i.x)
    return max_weight_solutions

def get_solution_set(n, A_eq, b_eq, A_ub, b_ub, bounds, LP_json, n_directions):
    directions = sample_directions(n+1, int(n_directions/2), kind="rademacher")
    directions.extend(sample_directions(n+1, int(n_directions/2), kind="gaussian"))
    sols = get_max_weight_solution(n, A_eq, b_eq, A_ub, b_ub, bounds, LP_json)
    # print (f"Number of max weight solutions: {len(sols)}")
    for d in directions:
        # minimize d^T w
        res_min = linprog(d[:n],A_eq=A_eq, b_eq=b_eq,A_ub=A_ub, b_ub=b_ub,bounds=bounds)
        if res_min.success:
            sols.append(res_min.x)

        # maximize d^T w == minimize (-d)^T w
        res_max = linprog(-d[:n],A_eq=A_eq, b_eq=b_eq,A_ub=A_ub, b_ub=b_ub,bounds=bounds)
        if res_max.success:
            sols.append(res_max.x)

        res_min = linprog(d,A_eq=LP_json["A_eq"], b_eq=LP_json["b_eq"],A_ub=LP_json["A_ub"], b_ub=LP_json["b_ub"],bounds=LP_json["bounds"])
        if res_min.success:
            sols.append(res_min.x[:n])

        res_max = linprog(-d,A_eq=LP_json["A_eq"], b_eq=LP_json["b_eq"],A_ub=LP_json["A_ub"], b_ub=LP_json["b_ub"],bounds=LP_json["bounds"])
        if res_max.success:
            sols.append(res_max.x[:n])
  
    return sols

def find_most_different_solutions(sols, expected_freqs):
    solution_combinations = list(itertools.combinations(sols, 2))

    max_diff = 0
    max_diff_solutions = None

    max_inf_diff = 0
    max_inf_diff_solutions = None
    for i, j in solution_combinations:
        assert np.isclose(np.dot(i, expected_freqs), HOURS_PER_DAY)
        assert np.isclose(np.dot(j, expected_freqs), HOURS_PER_DAY)
        #element wise multiplication of i and j with expected_freqs
        i_weighted = i * expected_freqs
        j_weighted = j * expected_freqs
        diff = np.linalg.norm(i_weighted - j_weighted)
        if diff > max_diff:
            max_diff = diff
            max_diff_solutions = (i_weighted, j_weighted)

        inf_diff = np.linalg.norm(i_weighted - j_weighted, ord=np.inf)
        if inf_diff > max_inf_diff:
            max_inf_diff = inf_diff
            max_inf_diff_solutions = (i_weighted, j_weighted)

        
    return max_diff, max_diff_solutions, max_inf_diff, max_inf_diff_solutions

if __name__ == "__main__":

    #get soc code indices as command line arguments via argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument("--SOC_code", type=str, default="")
    parser.add_argument("--extra_details", type=str, default="")
    parser.add_argument("--use_chosen_codes", action="store_true", default=False)
    parser.add_argument("--print_all_details", action="store_true", default=False)
    parser.add_argument("--use_max_span_time_weights", action="store_true", default=True)
    # parser.add_argument("--n_directions", type=int, default=100)
    args = parser.parse_args()
    # SOC_code = args.SOC_code
    # extra_details = args.extra_details
    # n_directions = args.n_directions

    extra_details=args.extra_details
    max_span_suffix = "_max_span_weigths" if args.use_max_span_time_weights else ""

    # extra_details += "_recompute_agreement_less_conservative"


    n_directions = 1000
    # WORKBank_soc_codes = ['43-4051.00', '11-9111.00', '17-2141.00', '27-3041.00', '15-1231.00', '15-1251.00', '15-1299.09', '41-4011.00', '15-1299.08', '13-1071.00', '43-6014.00', '11-9121.01', '15-1232.00', '13-2071.00', '11-3021.00', '15-1211.00', '13-1199.05', '43-3031.00', '13-2061.00', '13-1111.00', '25-1011.00', '13-2011.00', '43-9041.00', '43-5061.00', '15-1253.00', '11-9199.01', '15-1254.00', '23-1011.00', '27-2012.05', '13-2099.01', '11-3061.00', '27-3023.00', '29-1224.00', '51-8012.00', '17-3011.00', '15-1212.00', '17-3027.00', '11-3051.03', '13-2023.00', '11-3121.00', '17-2171.00', '27-3042.00', '43-9081.00', '43-9031.00', '13-1081.02', '13-2052.00', '11-3071.04', '15-2051.02', '13-1161.01', '17-1011.00', '41-3031.00', '13-2031.00', '27-3043.00', '15-2051.01', '13-2041.00', '13-2072.00', '31-9094.00', '43-6012.00', '43-6013.00', '17-2051.00', '11-2011.00', '19-3099.01', '13-1199.06', '13-1023.00', '41-3041.00', '43-4181.00', '43-9021.00', '11-3051.01', '15-1255.01', '13-1151.00', '27-1024.00', '17-2011.00', '43-5031.00', '13-2082.00', '23-1012.00', '27-4021.00', '15-1211.01', '27-2012.00', '27-1011.00', '11-3031.00', '25-4022.00', '15-1244.00', '19-1029.01', '15-1221.00', '13-1131.00', '41-9041.00', '27-3031.00', '13-1051.00', '19-1029.02', '13-1041.00', '15-1242.00', '15-2041.00', '15-2041.01', '11-3031.01', '43-3051.00', '15-1299.01', '19-4061.00', '15-2021.00', '19-4099.01', '17-2199.03', '19-4012.01', '43-4061.00', '19-3092.00', '43-4031.00']
    if args.use_chosen_codes:
        with open(f"data/generated_data/soc_code_dones_{extra_details}.txt", "r") as f:
            processed_codes = [line.strip() for line in f.readlines()]
    #    WORKBank_soc_codes = ["27-3043.00", "13-2031.00", "11-9199.01", "13-2099.01"]
       #['43-4051.00', '11-9111.00', '17-2141.00', '27-3041.00', '15-1231.00', '15-1251.00', '15-1299.09', '41-4011.00', '15-1299.08', '13-1071.00', '43-6014.00', '11-9121.01', '15-1232.00', '13-2071.00', '11-3021.00', '15-1211.00', '13-1199.05', '43-3031.00', '13-2061.00', '13-1111.00', '25-1011.00', '13-2011.00', '43-9041.00', '43-5061.00', '15-1253.00', '11-9199.01', '15-1254.00', '23-1011.00', '27-2012.05', '13-2099.01', '11-3061.00', '27-3023.00', '29-1224.00', '51-8012.00', '17-3011.00', '15-1212.00', '17-3027.00', '11-3051.03', '13-2023.00', '11-3121.00', '17-2171.00', '27-3042.00', '43-9081.00', '43-9031.00', '13-1081.02', '13-2052.00', '11-3071.04', '15-2051.02', '13-1161.01', '17-1011.00', '41-3031.00', '13-2031.00', '27-3043.00', '15-2051.01', '13-2041.00', '13-2072.00', '31-9094.00', '43-6012.00', '43-6013.00', '17-2051.00', '11-2011.00', '19-3099.01', '43-4181.00', '43-9021.00', '11-3051.01', '15-1255.01', '13-1151.00']
    # else:
    #     WORKBank_soc_codes = WORKBank_soc_codes[0:50]
    biggest_task_diffs = []
    biggest_task_diffs_from_original = []

    l2_diffs = []
    l2_diffs_from_original = []
    
    for SOC_code in processed_codes:
    
        print (f"Testing with O*NET-SOC Code: {SOC_code}")
        #load the LP json file
        try:
            with open(f"data/generated_data/occupation_time_per_task_{SOC_code}{extra_details}_chosen_LP{max_span_suffix}.json", "r") as f:
                LP_json = json.load(f)
        except:
            print (f"***New data format not saved, skipping {SOC_code}***\n")
            continue

        #load the occupation time per task json file
        with open(f"data/generated_data/occupation_time_per_task_{SOC_code}{extra_details}_chosen{max_span_suffix}.json", "r") as f:
            occupation_time_per_task = json.load(f)

        expected_freqs = np.array(list(occupation_time_per_task["Expected freq per task"].values()))

        #find original weights
        # res=linprog(LP_json["c"],A_eq=LP_json["A_eq"], b_eq=LP_json["b_eq"],A_ub=LP_json["A_ub"], b_ub=LP_json["b_ub"],bounds=LP_json["bounds"])
        original_weights = np.array(list(occupation_time_per_task["Time per task"].values()))
        task_codes = list(occupation_time_per_task["Time per task"].keys())
       
        different_res=linprog(-np.array(LP_json["c"][:-1]),A_eq=np.array(LP_json["A_eq"])[:, :-1], b_eq=np.array(LP_json["b_eq"]),A_ub=np.array(LP_json["A_ub"])[:, :-1], b_ub=np.array(LP_json["b_ub"]),bounds=np.array(LP_json["bounds"])[:-1, :]) 
        different_weights = different_res.x

        original_weights_weighted = original_weights * expected_freqs
        different_weights_weighted = different_weights * expected_freqs
        max_diff_from_original = np.linalg.norm(
            original_weights_weighted - different_weights_weighted
        )
        max_diff_from_original_weights = different_weights_weighted

        max_inf_diff_from_original = np.linalg.norm(
            original_weights_weighted - different_weights_weighted, ord=np.inf
        )
        max_inf_diff_from_original_weights = different_weights_weighted

        #first, lets remove the constraints where we maximize the min value of w
        c = np.zeros(len(LP_json["c"])-1)
        # c = np.array(LP_json["c"])
        A_ub = np.array(LP_json["A_ub"])
        b_ub = np.array(LP_json["b_ub"])

        A_eq = np.array(LP_json["A_eq"])
        b_eq = np.array(LP_json["b_eq"])

        A_eq = A_eq[:, :-1]
        A_ub = A_ub[:, :-1]

        n = len(A_ub[0])
        assert len(original_weights) == n
        bounds = LP_json["bounds"][:-1]

        sols = get_solution_set(n, A_eq, b_eq, A_ub, b_ub, bounds, LP_json, n_directions)
        sols.append(original_weights)
        sols.append(different_weights)
            
        max_diff, max_diff_solutions, max_inf_diff, max_inf_diff_solutions = find_most_different_solutions(sols, expected_freqs)
        print (f"Number of solutions found: {len(sols)}")
        #loop through all combinations of solutions and compute the difference

        for sol in sols:
            sol_weighted = sol * expected_freqs
            diff = np.linalg.norm(sol_weighted - original_weights_weighted)
            if diff > max_diff_from_original:
                max_diff_from_original = diff
                max_diff_from_original_weights = sol_weighted

            inf_diff = np.linalg.norm(sol_weighted - original_weights_weighted, ord=np.inf)
            if inf_diff > max_inf_diff_from_original:
                max_inf_diff_from_original = inf_diff
                max_inf_diff_from_original_weights = sol_weighted

        print (f"Max difference between solutions (norm): {max_diff}")
        print (f"Max difference between solutions (norm inf): {max_inf_diff}")
        print (f"Max difference between found solutions and original weights (norm): {max_diff_from_original}")
        print (f"Max difference between found solutions and original weights (norm inf): {max_inf_diff_from_original}")

        if args.print_all_details:
            #for each task code, print the corrosponing original weight and weight from max_inf_diff solutions
            for code_i, code in enumerate(task_codes):
                print (f"Task code: {code}")
                print (f"Original predicted time per task per day: {original_weights_weighted[code_i]}")
                print (f"Predicted time per task per day from max_inf_diff solutions: {max_inf_diff_from_original_weights[code_i]}")
                print ("--------------------------------")
            print ("\n--------------------------------\n")
            for code_i, code in enumerate(task_codes):
                print (f"Task code: {code}")
                print (f"Original predicted time per task per day: {original_weights_weighted[code_i]}")
                print (f"Predicted time per task per day from max_diff solutions: {max_diff_from_original_weights[code_i]}")
                print ("--------------------------------")
            print ("\n--------------------------------\n")
       
        biggest_task_diffs.append(max_inf_diff)
        biggest_task_diffs_from_original.append(max_inf_diff_from_original)

        l2_diffs.append(max_diff)
        l2_diffs_from_original.append(max_diff_from_original)

    print (f"Mean biggest task time per task per day difference: {np.mean(biggest_task_diffs)}")
    print (f"Std biggest task time per task per day difference: {np.std(biggest_task_diffs)}")
    print (f"Max biggest task time per task per day difference: {np.max(biggest_task_diffs)}")
    print (f"Min biggest task time per task per day difference: {np.min(biggest_task_diffs)}")
    print ("\n--------------------------------\n")
    print (f"Mean biggest task time per task per day difference from original weights: {np.mean(biggest_task_diffs_from_original)}")
    print (f"Std biggest task time per task per day difference from original weights: {np.std(biggest_task_diffs_from_original)}")
    print (f"Max biggest task time per task per day difference from original weights: {np.max(biggest_task_diffs_from_original)}")
    print (f"Min biggest task time per task per day difference from original weights: {np.min(biggest_task_diffs_from_original)}")


    print (f"Mean biggest L2 difference between solutions: {np.mean(l2_diffs)}")
    print (f"Std biggest L2 difference between solutions: {np.std(l2_diffs)}")
    print (f"Max biggest L2 difference between solutions: {np.max(l2_diffs)}")
    print (f"Min biggest L2 difference between solutions: {np.min(l2_diffs)}")

    print (f"Mean biggest L2 difference from original weights: {np.mean(l2_diffs_from_original)}")
    print (f"Std biggest L2 difference from original weights: {np.std(l2_diffs_from_original)}")
    print (f"Max biggest L2 difference from original weights: {np.max(l2_diffs_from_original)}")
    print (f"Min biggest L2 difference from original weights: {np.min(l2_diffs_from_original)}")