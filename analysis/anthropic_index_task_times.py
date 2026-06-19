
import pandas as pd
import re


def get_anthropic_task_time_mapping():
    # See documentation here:
    # https://huggingface.co/datasets/Anthropic/EconomicIndex/blob/main/release_2026_01_15/data_documentation.md
    df = pd.read_csv("data/Anthropic_Index/aei_raw_claude_ai_2025-11-13_to_2025-11-20.csv")

    # Extract onet_task::human_only_time (global only) and build a task->time mapping.
    facet_df = df[df["facet"] == "onet_task::human_only_time"]
    duplicate_keys = (
        facet_df.groupby(["cluster_name", "variable"])
        .size()
        .reset_index(name="count")
    )
    duplicate_rows = duplicate_keys[duplicate_keys["count"] > 1]
    if not duplicate_rows.empty:
        raise ValueError(
            "Found multiple entries for the same task and variable. "
            f"Example: {duplicate_rows.head(5).to_dict(orient='records')}"
        )

    task_time_pivot = facet_df.pivot(
        index="cluster_name", columns="variable", values="value"
    )
    required_columns = [
        "onet_task_human_only_time_mean",
        "onet_task_human_only_time_mean_ci_lower",
        "onet_task_human_only_time_mean_ci_upper",
    ]
    missing_columns = [
        column for column in required_columns if column not in task_time_pivot.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    task_time_mapping = (
        task_time_pivot[required_columns]
        .rename(
            columns={
                "onet_task_human_only_time_mean": "human_only_time_mean",
                "onet_task_human_only_time_mean_ci_lower": "human_only_time_mean_ci_lower",
                "onet_task_human_only_time_mean_ci_upper": "human_only_time_mean_ci_upper",
            }
        )
        .reset_index()
        .rename(columns={"cluster_name": "task"})
        .sort_values("task")
        .reset_index(drop=True)
    )

    def normalize_task_text(text):
        if pd.isna(text):
            return ""
        normalized = re.sub(r"\s+", " ", str(text).strip().lower())
        return normalized


    # Map tasks to O*NET-SOC occupation codes via Task Statements.xlsx
    task_statements = pd.read_excel("data/Task Statements.xlsx")
    task_statements["task_norm"] = task_statements["Task"].apply(normalize_task_text)
    task_time_mapping["task_norm"] = task_time_mapping["task"].apply(normalize_task_text)


    task_time_with_soc = task_time_mapping.merge(
        task_statements[
            ["O*NET-SOC Code", "Title", "Task ID", "Task", "task_norm"]
        ].drop_duplicates(),
        on="task_norm",
        how="left",
    )

    #print number of tasks in task_time_with_soc
    print (f"Number of tasks in task_time_with_soc: {len(task_time_with_soc)}")
    #print total number of tasks in ONET

    return task_time_with_soc
