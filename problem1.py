"""
Problem 1: Log Level Distribution
---------------------------------
Analyze the distribution of log levels (INFO, WARN, ERROR, DEBUG)
across all Spark log files and output statistics.

Outputs:
1. data/output/problem1_counts.csv  -> log level counts
2. data/output/problem1_sample.csv  -> 10 random sample log entries
3. data/output/problem1_summary.txt -> summary statistics
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, regexp_extract, input_file_name, to_timestamp
)
from pathlib import Path
import argparse

# spark Session Helper
def get_spark(app_name: str, master_url: str = None):
    builder = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                "com.amazonaws.auth.InstanceProfileCredentialsProvider")
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
    )

    if master_url:
        print(f"Connecting to Spark cluster at {master_url}")
        builder = builder.master(master_url)
    else:
        print("Running in local mode (local[*])")
        builder = builder.master("local[*]")

    spark = builder.getOrCreate()
    print(f"Spark master: {spark.sparkContext.master}")
    return spark

# main logic
def run_problem1(input_path: str, output_dir: str, master_url: str = None):
    spark = get_spark("Problem1_LogLevelDistribution", master_url)
    sc = spark.sparkContext
    sc._jsc.hadoopConfiguration().setBoolean(
        "mapreduce.input.fileinputformat.input.dir.recursive", True
    )

    print(f"Reading logs recursively from: {input_path}")
    logs_df = spark.read.text(input_path).withColumn("file_path", input_file_name())
    total_lines = logs_df.count()
    print(f"Total lines read: {total_lines:,}")

    # filter out system messages (not starting with timestamp)
    logs_df = logs_df.filter(col("value").rlike(r"^\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}"))
    total_lines_filtered = logs_df.count()
    print(f"Total lines read after filtering: {total_lines_filtered:,}")

    # parse using regexp_extract
    logs_parsed = logs_df.select(
        regexp_extract("value", r"^(\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})", 1).alias("timestamp"),
        regexp_extract("value", r"(INFO|WARN|ERROR|DEBUG)", 1).alias("log_level"),
        regexp_extract("value", r"(INFO|WARN|ERROR|DEBUG)\s+([^\s]+)", 2).alias("component"),
        col("value").alias("message"),
        col("file_path")
    )

    # clean and handle timestamp(make string to time)
    logs_parsed = logs_parsed.withColumn(
        "timestamp", to_timestamp("timestamp", "yy/MM/dd HH:mm:ss")
    )

    # filter valid log levels
    logs_valid = logs_parsed.filter(col("log_level").isNotNull() & (col("log_level") != ""))
    logs_valid.cache()  # optimization
    valid_lines = logs_valid.count()
    print(f"Lines with valid log levels: {valid_lines:,}")

    # count per log level
    counts_df = logs_valid.groupBy("log_level").count().orderBy(col("count").desc())
    counts_path = str(Path(output_dir) / "problem1_counts.csv")
    counts_df.coalesce(1).write.csv(counts_path, header=True, mode="overwrite")
    print(f"Saved counts to {counts_path}")

    # random sample (10)
    sample_df = logs_valid.orderBy(col("timestamp")).limit(10)

    sample_df = sample_df.select(
        col("message").alias("log_entry"),
        col("log_level")
    )

    sample_path = str(Path(output_dir) / "problem1_sample.csv")
    sample_df.coalesce(1).write.csv(sample_path, header=True, mode="overwrite")
    print(f"Saved 10 sample logs to {sample_path}")

    # write summary
    counts_collect = counts_df.collect()
    total_with_levels = sum([r["count"] for r in counts_collect])
    unique_levels = len(counts_collect)

    summary_lines = [
        f"Total log lines processed: {total_lines:,}",
        f"Total lines with log levels: {total_with_levels:,}",
        f"Unique log levels found: {unique_levels}\n",
        "Log level distribution:",
    ]
    for row in counts_collect:
        pct = row["count"] / total_with_levels * 100
        summary_lines.append(f"  {row['log_level']:<7}: {row['count']:>10,} ({pct:5.2f}%)")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    summary_path = str(Path(output_dir) / "problem1_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"Saved summary to {summary_path}")

    spark.stop()
    print("Problem 1 completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Problem 1: Log Level Distribution")
    parser.add_argument("--net-id", type=str, help="Your Georgetown NetID (e.g., rx67)")
    parser.add_argument("--input_path", type=str, default=None, help="S3 or local input path")
    parser.add_argument("--output_dir", type=str, default=None, help="S3 or local output directory")
    parser.add_argument("--master_url", type=str, default=None, help="Spark master URL")
    args = parser.parse_args()

    if args.net_id:
        s3_prefix = f"s3a://{args.net_id}-assignment-spark-cluster-logs"
        if not args.input_path:
            args.input_path = f"{s3_prefix}/data/"
        if not args.output_dir:
            args.output_dir = f"{s3_prefix}/output/"
        print(f"Using S3 paths from NetID '{args.net_id}':")
        print(f"Input:  {args.input_path}")
        print(f"Output: {args.output_dir}")

    run_problem1(args.input_path, args.output_dir, args.master_url)
