import json
import numpy as np
import argparse
from scipy.stats import kendalltau
from collections import defaultdict
from scipy import stats
import matplotlib.pyplot as plt
from utils.constants import (
    FREQ_TO_TIME_PER_DAY,
    HOURS_PER_DAY,
)
from utils.compute_onet_hours import compute_soc_constants
from find_time_per_task import get_onet_task_data
from analysis.anthropic_index_task_times import get_anthropic_task_time_mapping


if __name__ == "__main__":

    #load extra_details and 
    parser = argparse.ArgumentParser()
    # parser.add_argument("--SOC_code", type=str, default="")
    parser.add_argument("--extra_details", type=str, default="")
    parser.add_argument("--use_chosen_codes", action="store_true", default=False)
    parser.add_argument("--use_cps_constants", action="store_true", default=False)
    parser.add_argument("--tuning_params", action="store_true", default=False)
    parser.add_argument("--use_uniform_baseline", action="store_true", default=False)
    parser.add_argument("--use_max_span_time_weights", action="store_true", default=True)

    #python3 -m analysis.time_share_correlations --use_cps_constants --use_chosen_codes --extra_details  gpt-5.2_ONET_30.2_use_cps_constants_prob_thresh=0.7 --use_max_span_time_weights
    

    # parser.add_argument("--n_directions", type=int, default=100)
    args = parser.parse_args()
    # SOC_code = args.SOC_code
    # extra_details = args.extra_details
    # n_directions = args.n_directions

    extra_details=args.extra_details
    max_span_suffix = "_max_span_weigths" if args.use_max_span_time_weights else ""
    if args.use_cps_constants:
        assert "_use_cps_constants" in extra_details, "extra_details must contain _use_cps_constants"
    #     extra_details += "_use_cps_constants"

    # processed_codes = ['43-4051.00', '11-9111.00', '17-2141.00', '27-3041.00', '15-1231.00', '15-1251.00', '15-1299.09', '41-4011.00', '15-1299.08', '13-1071.00', '43-6014.00', '11-9121.01', '15-1232.00', '13-2071.00', '11-3021.00', '15-1211.00', '13-1199.05', '43-3031.00', '13-2061.00', '13-1111.00', '25-1011.00']
    WORKBank_soc_codes = ['43-4051.00', '11-9111.00', '17-2141.00', '27-3041.00', '15-1231.00', '15-1251.00', '15-1299.09', '41-4011.00', '15-1299.08', '13-1071.00', '43-6014.00', '11-9121.01', '15-1232.00', '13-2071.00', '11-3021.00', '15-1211.00', '13-1199.05', '43-3031.00', '13-2061.00', '13-1111.00', '25-1011.00', '13-2011.00', '43-9041.00', '43-5061.00', '15-1253.00', '11-9199.01', '15-1254.00', '23-1011.00', '27-2012.05', '13-2099.01', '11-3061.00', '27-3023.00', '29-1224.00', '51-8012.00', '17-3011.00', '15-1212.00', '17-3027.00', '11-3051.03', '13-2023.00', '11-3121.00', '17-2171.00', '27-3042.00', '43-9081.00', '43-9031.00', '13-1081.02', '13-2052.00', '11-3071.04', '15-2051.02', '13-1161.01', '17-1011.00', '41-3031.00', '13-2031.00', '27-3043.00', '15-2051.01', '13-2041.00', '13-2072.00', '31-9094.00', '43-6012.00', '43-6013.00', '17-2051.00', '11-2011.00', '19-3099.01', '13-1199.06', '13-1023.00', '41-3041.00', '43-4181.00', '43-9021.00', '11-3051.01', '15-1255.01', '13-1151.00', '27-1024.00', '17-2011.00', '43-5031.00', '13-2082.00', '23-1012.00', '27-4021.00', '15-1211.01', '27-2012.00', '27-1011.00', '11-3031.00', '25-4022.00', '15-1244.00', '19-1029.01', '15-1221.00', '13-1131.00', '41-9041.00', '27-3031.00', '13-1051.00', '19-1029.02', '13-1041.00', '15-1242.00', '15-2041.00', '15-2041.01', '11-3031.01', '43-3051.00', '15-1299.01', '19-4061.00', '15-2021.00', '19-4099.01', '17-2199.03', '19-4012.01', '43-4061.00', '19-3092.00', '43-4031.00']
    # processed_codes = WORKBank_soc_codes[0:50]
    if args.tuning_params:
        from utils.codes import PARAM_TUNING_SOC_CODES
        print (f"Using {len(PARAM_TUNING_SOC_CODES)} tuning parameters")
        processed_codes = PARAM_TUNING_SOC_CODES
    elif args.use_chosen_codes:
        # processed_codes = ["13-2041.00"]
        with open(f"data/soc_code_dones_{extra_details}.txt", "r") as f:
            processed_codes = [line.strip() for line in f.readlines()]
        # processed_codes = ['43-4051.00', '11-9111.00', '17-2141.00', '27-3041.00', '15-1231.00', '15-1251.00', '15-1299.09', '41-4011.00', '15-1299.08', '13-1071.00', '43-6014.00', '11-9121.01', '15-1232.00', '13-2071.00', '11-3021.00', '15-1211.00', '13-1199.05', '43-3031.00', '13-2061.00', '13-1111.00', '25-1011.00', '13-2011.00', '43-9041.00', '43-5061.00', '15-1253.00', '11-9199.01', '15-1254.00', '23-1011.00', '27-2012.05', '13-2099.01', '11-3061.00', '27-3023.00', '29-1224.00', '51-8012.00', '17-3011.00', '15-1212.00', '17-3027.00', '11-3051.03', '13-2023.00', '11-3121.00', '17-2171.00', '27-3042.00', '43-9081.00', '43-9031.00', '13-1081.02', '13-2052.00', '11-3071.04', '15-2051.02', '13-1161.01', '17-1011.00', '41-3031.00', '13-2031.00', '27-3043.00', '15-2051.01', '13-2041.00', '13-2072.00', '31-9094.00', '43-6012.00', '43-6013.00', '17-2051.00', '11-2011.00', '19-3099.01', '43-4181.00', '43-9021.00', '11-3051.01', '15-1255.01', '13-1151.00']
    else:
        processed_codes = WORKBank_soc_codes[0:50]

#    extra_details

    correlations = []
    p_values = []
    stat_pos_corrs = []
    stat_pos_corrs_is_core = []

    stat_sig_freq_corrs = []

    anthropic_stat_pos_corrs = []
    anthropic_n_uncovered_occupations = 0
    anthropic_pairwise_violations =[]
    anthropic_pairwise_comparisons =[]
    anthropic_task_time_mapping = get_anthropic_task_time_mapping()
    
    task2pred_time_per_day = defaultdict(list)
    task2pred_time = defaultdict(list)
    task2anthropic_pred_time_per_day = defaultdict(list)

    data2save = {}

    n_anthropic_times_above_daily_hours = 0
    n_anthropic_times_and_freq_above_daily_hours = 0
    n_occupations_with_atleast_two_anthropic_preds = 0
    
    for ONET_SOC_CODE in processed_codes:
        
        print (f"Testing with O*NET-SOC Code: {ONET_SOC_CODE}")
        try:
            with open(
                f"data/generated_data/occupation_time_per_task_{ONET_SOC_CODE}{extra_details}_chosen{max_span_suffix}.json",
                "r",
            ) as f:
                occupation_result = json.load(f)
        except:
            print (f"***No data found for O*NET-SOC Code: {ONET_SOC_CODE}***\n")
            continue
        # print (occupation_result)
        # with open(f"data/eval_data/task_data_{ONET_SOC_CODE}.json", "r") as f:
        #     task_data = json.load(f)
        # print (task_data)
        task_data, task_statements = get_onet_task_data(ONET_SOC_CODE, return_statements=True)
        freq_to_time_per_day = FREQ_TO_TIME_PER_DAY
        if args.use_cps_constants:
            cps_constants = compute_soc_constants(ONET_SOC_CODE)
            freq_to_time_per_day = cps_constants["FREQ_TO_TIME_PER_DAY"]
        

        #get the tasks in task_data that have the same occupation title
        task_data = task_data["Tasks"]
        task_codes = [str(task["Task ID"]) for task in task_data]
        task_titles = [task["Task"] for task in task_data]
        importance = [task["Ratings"]["Importance"]["Data Value"] for task in task_data]
        expected_frequency = []
        for task in task_data:
            task_frequency =  task["Ratings"]["Frequency"]
            expected_freq = sum(freq_to_time_per_day[freq["Category"]]*(freq["Data Value"]/100) for freq in task_frequency)
            expected_frequency.append(expected_freq)
        
        # improtance_times_expected_freq = [importance[i] * expected_frequency[i] for i in range(len(importance))]


        #get anthropic time for this occupation
        anthropic_time = anthropic_task_time_mapping[anthropic_task_time_mapping["O*NET-SOC Code"] == ONET_SOC_CODE]

        # print (anthropic_time["Task ID"])
        # print (task_codes)

        predicted_times_per_task = []
        predicted_times_per_day_per_task = []
        predicted_times_per_task_anthropic_paired = []
        predicted_times_per_day_per_task_anthropic_paired = []
        is_core_predicted_time_per_day = []
        is_core_improtance_times_expected_freq = []
        

        anthropic_times = []
        anthropic_times_times_expected_freq = []
        anthropic_times_upper_ci = []
        anthropic_times_lower_ci = []
        for task_code_i, task_code in enumerate(task_codes):
            predicted_time = occupation_result["Time per task"][task_code]
            expected_freq = occupation_result["Expected freq per task"][task_code]
            if args.use_uniform_baseline:
                predicted_time_per_day = 1/len(task_codes)
            else:
                predicted_time_per_day = predicted_time * expected_freq
            predicted_times_per_task.append(predicted_time)
            predicted_times_per_day_per_task.append(predicted_time_per_day)
            task2pred_time_per_day[task_code].append(predicted_time_per_day)
            task2pred_time[task_code].append(predicted_time)
            #get task_code row in task_statements
            task_statement_row = task_statements[task_statements["Task ID"] == int(task_code)]
            assert len(task_statement_row) == 1
            is_core = task_statement_row["Task Type"].values[0] == "Core"
            if is_core:
                is_core_predicted_time_per_day.append(predicted_time_per_day)
                is_core_improtance_times_expected_freq.append(importance[task_code_i] * expected_freq)
            # if task_statements[task_code_i]["Task Type"] == "Core":
            #     is_core_predicted_time_per_day.append(predicted_time * expected_freq)
            #     is_core_improtance_times_expected_freq.append(importance[task_code_i] * expected_freq)
            #check if task_code is in anthropic_time
            if int(task_code) in anthropic_time["Task ID"].values:
                anthropic_rows = anthropic_time[anthropic_time["Task ID"] == int(task_code)]
                assert len(anthropic_rows) == 1, (
                    f"Expected exactly one Anthropic row per (SOC, Task ID); got {len(anthropic_rows)} for {ONET_SOC_CODE}, {task_code}"
                )
                anthropic_pred_time = anthropic_rows["human_only_time_mean"].values[0]
                pred_time_upper_ci = anthropic_rows["human_only_time_mean_ci_upper"].values[0]
                pred_time_lower_ci = anthropic_rows["human_only_time_mean_ci_lower"].values[0]

                anthropic_times.append(anthropic_pred_time)
                anthropic_times_times_expected_freq.append(anthropic_pred_time * expected_freq)
                anthropic_times_upper_ci.append(pred_time_upper_ci)
                anthropic_times_lower_ci.append(pred_time_lower_ci)
                predicted_times_per_day_per_task_anthropic_paired.append(predicted_time_per_day)
                predicted_times_per_task_anthropic_paired.append(predicted_time)

                task2anthropic_pred_time_per_day[task_code].append((anthropic_pred_time, pred_time_lower_ci, pred_time_upper_ci))
        
        #compute pairwise accuracy between predicted_times_per_task_anthropic_paired and anthropic_times
        pairwise_accuracy = 0
        n_comparisons = 0
        for i, predicted_time_i in enumerate(predicted_times_per_task_anthropic_paired):
            for j, predicted_time_j in enumerate(predicted_times_per_task_anthropic_paired[i + 1 :]):
                j_full = i + 1 + j  # index in full list (inner loop iterates over slice)
                if predicted_time_i > predicted_time_j and anthropic_times_lower_ci[i] > anthropic_times_upper_ci[j_full]:
                    pairwise_accuracy += 1
                elif predicted_time_i < predicted_time_j and anthropic_times_upper_ci[i] < anthropic_times_lower_ci[j_full]:
                    pairwise_accuracy += 1

                if anthropic_times_lower_ci[i] > anthropic_times_upper_ci[j_full] or anthropic_times_upper_ci[i] < anthropic_times_lower_ci[j_full]:
                    n_comparisons += 1

        #compute KT correlation coefficient between predicted_times_per_day_per_task and mean_times
        kt_correlation, kt_p_value = kendalltau(predicted_times_per_day_per_task, importance)
        print (f"KT correlation coefficient between predicted times per day per task and ONET importances: {kt_correlation}")
        print (f"KT p-value: {kt_p_value}")
        print ('\n')

        kt_correlation_is_core_predicted_time_per_day, kt_p_value_is_core_predicted_time_per_day = kendalltau(is_core_predicted_time_per_day, is_core_improtance_times_expected_freq)
        print (f"KT correlation coefficient between is core predicted time per day and is core importance times expected frequency: {kt_correlation_is_core_predicted_time_per_day}")
        print (f"KT p-value: {kt_p_value_is_core_predicted_time_per_day}")
        print ('\n')
        # kt_correlation_anthropic_paired, kt_p_value_anthropic_paired = kendalltau(predicted_times_per_day_per_task_anthropic_paired, anthropic_times)
        # print (f"KT correlation coefficient between predicted times per day per task and ONET importances (anthropic paired): {kt_correlation_anthropic_paired}")
        # print (f"KT p-value: {kt_p_value_anthropic_paired}")
        # print (f"Fraction of coverage with anthropic paired: {len(predicted_times_per_day_per_task_anthropic_paired)}/{len(predicted_times_per_day_per_task)}")
        # print ('\n')
        kt_correlation_anthropic_paired_predicted_times_per_task, kt_p_value_anthropic_paired_predicted_times_per_task = kendalltau(predicted_times_per_task_anthropic_paired, anthropic_times)
        print (f"KT correlation coefficient between predicted times per task and Anthropic time estimates: {kt_correlation_anthropic_paired_predicted_times_per_task}")
        print (f"KT p-value: {kt_p_value_anthropic_paired_predicted_times_per_task}")
        print (f"Fraction of coverage with anthropic paired: {len(predicted_times_per_task_anthropic_paired)}/{len(predicted_times_per_task)}")
        print ('\n')
        # kt_correlation_importance_anthropic_times, kt_p_value_importance_anthropic_times = kendalltau(importance_paired, anthropic_times)
        # print (f"KT correlation coefficient between importance and ONET importances (anthropic paired): {kt_correlation_importance_anthropic_times}")
        # print (f"KT p-value: {kt_p_value_importance_anthropic_times}")
        # print ('\n')
        #kt correlation between frequency and predicted_times_per_task
        kt_correlation_freq_predicted_time, kt_p_value_freq_predicted_time = kendalltau(expected_frequency, predicted_times_per_task)
        print (f"KT correlation coefficient between expected frequency and predicted times per task: {kt_correlation_freq_predicted_time}")
        print (f"KT p-value: {kt_p_value_freq_predicted_time}")
        print ('\n')

        if len(anthropic_times) >= 2:
            if np.sum(anthropic_times_times_expected_freq) > HOURS_PER_DAY:
                n_anthropic_times_and_freq_above_daily_hours += 1
            if np.sum(anthropic_times) > HOURS_PER_DAY:
                n_anthropic_times_above_daily_hours += 1
            n_occupations_with_atleast_two_anthropic_preds += 1

        if n_comparisons > 0:
            print (f"Pairwise accuracy between predicted times per task and anthropic times: {pairwise_accuracy}/{n_comparisons}")
            anthropic_pairwise_violations.append(1 - (pairwise_accuracy/n_comparisons))
            anthropic_pairwise_comparisons.append(n_comparisons)

        correlations.append(kt_correlation)
        p_values.append(kt_p_value)



        if kt_p_value_freq_predicted_time < 0.05:
            stat_sig_freq_corrs.append(kt_correlation_freq_predicted_time)
        #make sure kt_correlation is not nan
        if kt_correlation > 0 and kt_p_value < 0.05 and not np.isnan(kt_correlation):
            stat_pos_corrs.append(kt_correlation)

        if kt_correlation_is_core_predicted_time_per_day > 0 and kt_p_value_is_core_predicted_time_per_day < 0.05:
            stat_pos_corrs_is_core.append(kt_correlation_is_core_predicted_time_per_day)

        if np.isnan(kt_correlation):
            print ("WARNING: KT correlation is nan")
            print ((predicted_times_per_day_per_task, importance))

        if kt_correlation_anthropic_paired_predicted_times_per_task > 0 and kt_p_value_anthropic_paired_predicted_times_per_task < 0.05 and len(predicted_times_per_task_anthropic_paired)/len(predicted_times_per_task) > 0.7:
            anthropic_stat_pos_corrs.append(kt_correlation_anthropic_paired_predicted_times_per_task)
        if len(predicted_times_per_task_anthropic_paired)/len(predicted_times_per_task) > 0.7:
            anthropic_n_uncovered_occupations += 1

        entry = {
            "O*NET-SOC Code": ONET_SOC_CODE,
            "Title": occupation_result["Title"],
            "Task Titles": task_titles,
            "Task codes": task_codes,
            "Predicted times per task": predicted_times_per_task,
            "Predicted times per day per task": predicted_times_per_day_per_task
        }

        data2save[ONET_SOC_CODE] = entry

        print ("\n--------------------------------\n")

#save data2save to json
with open(f"data/analysis_results/occupation_data2saveforwebsite{extra_details}.json", "w") as f:
    json.dump(data2save, f, indent=4)

print (f"Mean correlation of predicted times per day per task and ONET importances: {np.mean(correlations)}")
print (f"Fraction of stat. sign. corrs: {len(stat_pos_corrs)}/{len(correlations)}")
print (f"Mean correlation of stat. sign. corrs: {np.mean(stat_pos_corrs)}")

print ("\n--------------------------------\n")
# print (stat_sig_freq_corrs)
print (f"Mean correlation of frequency and predicted times per task: {np.mean(stat_sig_freq_corrs)}")
print (f"Fraction of stat. sign. corrs between frequency and predicted times per task: {len(stat_sig_freq_corrs)}/{len(correlations)}")

print ("\n--------------------------------\n")
print (f"Mean correlation of is core predicted time per day and is core importance times expected frequency: {np.mean(stat_pos_corrs_is_core)}")
print (f"Fraction of stat. sign. corrs: {len(stat_pos_corrs_is_core)}/{len(correlations)}")
print (f"Mean correlation of stat. sign. corrs: {np.mean(stat_pos_corrs_is_core)}")

print ("\n--------------------------------\n")
print (f"Mean correlation of predicted times per task and Anhtropic time estimates (only when anthropic time estimates are available for more than 70% of the tasks): {np.mean(anthropic_stat_pos_corrs)}")
print (f"Fraction of stat. sign. corrs: {len(anthropic_stat_pos_corrs)}/{anthropic_n_uncovered_occupations}")
print (f"Mean # of ordering violations that our estimates satisfy of Anthropic time estimates (only when Anthropic estimates for two tasks have non-overlapping CIs, we compute the accuracy w.r.t that pair): {np.mean(anthropic_pairwise_violations)}")
print (f"Mean # of pairwise comparisons where two tasks have non-overlapping CIs, excluding occupations with 0: {np.mean(anthropic_pairwise_comparisons)}")
print ("Number of occupations with 0 non-overlapping pairs of tasks in Anthropic index:", n_occupations_with_atleast_two_anthropic_preds - len(anthropic_pairwise_comparisons))

print (f"Fraction of occupations with at least two Anthropic time estimates: {n_occupations_with_atleast_two_anthropic_preds}/{len(processed_codes)}")
print (f"Fraction of occupations with Anthropic time estimates * expected frequency above daily hours: {n_anthropic_times_and_freq_above_daily_hours}/{n_occupations_with_atleast_two_anthropic_preds}")
print (f"Fraction of occupations with Anthropic time estimates above daily hours: {n_anthropic_times_above_daily_hours}/{n_occupations_with_atleast_two_anthropic_preds}")
predicted_times_per_day2plot = []
predicted_times2plot = []
anthropic_times2plot = []
anthropic_lower2plot = []
anthropic_upper2plot = []
for task in task2anthropic_pred_time_per_day.keys():
    anthropic_pred_time, anthropic_lower_ci, anthropic_upper_ci = zip(*task2anthropic_pred_time_per_day[task])
    anthropic_mean = np.mean(anthropic_pred_time)
    anthropic_lower_mean = np.mean(anthropic_lower_ci)
    anthropic_upper_mean = np.mean(anthropic_upper_ci)
    anthropic_times2plot.append(anthropic_mean)
    anthropic_lower2plot.append(anthropic_lower_mean)
    anthropic_upper2plot.append(anthropic_upper_mean)
    predicted_times_per_day2plot.append(np.mean(task2pred_time_per_day[task]))
    predicted_times2plot.append(np.mean(task2pred_time[task]))
    # Task IDs are not globally unique: the same ID can denote different tasks in different occupations.
    # We key by task code only, so we assume each key here appears for exactly one occupation (one append per task).
    assert len(task2pred_time_per_day[task]) == len(task2pred_time[task]) == len(anthropic_pred_time) == len(anthropic_lower_ci) == len(anthropic_upper_ci) == 1

#save task2anthropic_pred_time_per_day and task2pred_time_per_day to json
with open(f"data/analysis_results/task2anthropic_pred_time_per_day{extra_details}.json", "w") as f:
    json.dump({task: {"anthropic_pred_time_per_day": task2anthropic_pred_time_per_day[task],
                     "predicted_time_per_day": task2pred_time_per_day[task],
                     "predicted_time": task2pred_time[task]} for task in task2anthropic_pred_time_per_day.keys()}, f, indent=4)

#compute spearmans correlation between predicted_times2plot and anthropic_times2plot
res = stats.spearmanr(predicted_times2plot, anthropic_times2plot)
print (f"Spearman's correlation between predicted times per task and anthropic times: {res.correlation}, p-value: {res.pvalue}")

# Plot: predicted time per day vs anthropic (with CI)
plt.figure(figsize=(8, 8))
if len(predicted_times_per_day2plot) > 0:
    x = np.array(predicted_times_per_day2plot)
    y = np.array(anthropic_times2plot)
    lower = np.array(anthropic_lower2plot)
    upper = np.array(anthropic_upper2plot)
    yerr = np.vstack([y - lower, upper - y])
    plt.errorbar(x, y, yerr=yerr, fmt='o', ecolor='gray', alpha=0.7, capsize=3)
    # add y=x reference and set axes to exact data min/max (no padding)
    minv = float(np.min(x))
    maxv = float(np.max(x))
    # use a tiny expansion only when all values are identical to avoid zero-range axes
    if minv == maxv:
        raise ValueError("All predicted times are identical; cannot plot.")
    else:
        lo = minv
        hi = maxv
    plt.plot([lo, hi], [lo, hi], 'k--', linewidth=1)
    # plt.xlim(lo, hi)
    # plt.ylim(lo, hi)
    # ticks = np.linspace(lo, hi, 5)
    # plt.xticks(ticks)
    # plt.yticks(ticks)
else:
    plt.scatter([], [])
plt.xlabel("Predicted time per task in hours (ours, averaged over occupations)")
plt.ylabel("Predicted time per task in hours (Anthropic)")
plt.savefig(f"data/analysis_results/anthropic_predicted_time_per_task_per_day_comparison{extra_details}.png")

# Plot: predicted time per task vs anthropic (with CI)
plt.figure(figsize=(8, 8))
if len(predicted_times2plot) > 0:
    x2 = np.array(predicted_times2plot)
    y2 = np.array(anthropic_times2plot)
    lower2 = np.array(anthropic_lower2plot)
    upper2 = np.array(anthropic_upper2plot)
    yerr2 = np.vstack([y2 - lower2, upper2 - y2])
    plt.errorbar(x2, y2, yerr=yerr2, fmt='o', ecolor='gray', alpha=0.7, capsize=3)
    # add y=x reference and set axes to exact data min/max (no padding)
    minv = float(np.min(x2))
    maxv = float(np.max(x2))
    if minv == maxv:
        raise ValueError("All predicted times are identical; cannot plot.")
    else:
        lo = minv
        hi = maxv
    plt.plot([lo, hi], [lo, hi], 'k--', linewidth=1)
    # plt.xlim(lo, hi)
    # plt.ylim(lo, hi)
    # ticks = np.linspace(lo, hi, 5)
    # plt.xticks(ticks)
    # plt.yticks(ticks)
else:
    plt.scatter([], [])
plt.xlabel("Predicted time per task in hours (ours, averaged over occupations)")
plt.ylabel("Predicted time per task in hours (Anthropic)")
plt.savefig(f"data/analysis_results/anthropic_predicted_time_per_task_comparison{extra_details}.png")


