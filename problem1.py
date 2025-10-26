"""
Problem 1: Log Level Distribution
---------------------------------
Analyze the distribution of log levels (INFO, WARN, ERROR, DEBUG)
across all Spark log files.

Outputs:
1. data/output/problem1_counts.csv  -> log level counts
2. data/output/problem1_sample.csv  -> 10 random sample log entries
3. data/output/problem1_summary.txt -> summary statistics
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit
import random
from pathlib import Path
import argparse

def run_problem1(input_path: str = "s3://rx67-assignment-spark-cluster-logs/data/",
                 output_dir: str = "data/output/"):
 
    # initialize Spark Session
    spark = SparkSession.builder.appName("Problem1_LogLevelDistribution").getOrCreate()
    sc = spark.sparkContext

    print(f"Reading logs from: {input_path}")
    logs_rdd = sc.textFile(input_path)

    total_lines = logs_rdd.count()
    print(f"Total log lines read: {total_lines:,}")

    # extract log levels 
    def extract_level(line):
        levels = ["INFO", "WARN", "ERROR", "DEBUG"]
        for lvl in levels:
            if f" {lvl} " in line:
                return (lvl, line)
        return (None, line)

    parsed_rdd = logs_rdd.map(extract_level)
    log_df = spark.createDataFrame(parsed_rdd, ["log_level", "log_entry"])

    # filter valid log levels
    filtered_df = log_df.filter(col("log_level").isNotNull())
    valid_lines = filtered_df.count()
    print(f"Lines with log levels: {valid_lines:,}")

    # compute counts per level
    counts_df = (
        filtered_df.groupBy("log_level")
        .count()
        .orderBy(col("count").desc())
    )

    # save to CSV
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    counts_path = output_dir + "problem1_counts.csv"
    counts_pd = counts_df.toPandas()
    counts_pd.to_csv(counts_path, index=False)
    print(f"Saved counts to {counts_path}")

    # sample 10 random log lines
    sample_rows = filtered_df.sample(False, 0.001).limit(10).collect()  # sample small fraction then take 10
    sample_output = "log_entry,log_level\n"
    for row in sample_rows:
        clean_entry = row["log_entry"].replace('"', '""')  # escape quotes
        sample_output += f"\"{clean_entry}\",{row['log_level']}\n"

    sample_path = output_dir + "problem1_sample.csv"
    with open(sample_path, "w") as f:
        f.write(sample_output)
    print(f"Saved sample logs to {sample_path}")

    # write summary statistics
    counts_dict = {row["log_level"]: row["count"] for row in counts_df.collect()}
    total_with_levels = sum(counts_dict.values())
    unique_levels = len(counts_dict)

    summary_lines = [
        f"Total log lines processed: {total_lines:,}",
        f"Total lines with log levels: {total_with_levels:,}",
        f"Unique log levels found: {unique_levels}\n",
        "Log level distribution:",
    ]

    for lvl, cnt in counts_dict.items():
        pct = cnt / total_with_levels * 100
        summary_lines.append(f"  {lvl:<7}: {cnt:>10,} ({pct:5.2f}%)")

    summary_path = output_dir + "problem1_summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))

    print(f"Saved summary to {summary_path}")

    # done
    print("Problem 1 completed successfully.")
    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Problem 1: Log Level Distribution")
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/sample/",
        help="Path to log file or directory (local or s3://)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/output/",
        help="Output directory for result files"
    )
    args = parser.parse_args()

    run_problem1(input_path=args.input_path, output_dir=args.output_dir)

