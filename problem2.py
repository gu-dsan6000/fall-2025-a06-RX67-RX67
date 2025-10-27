"""
Problem 2: Cluster Usage Analysis
---------------------------------
Analyze cluster usage patterns to understand which clusters
are most heavily used over time.

Extract cluster IDs, application IDs, and application start/end times
to create a time-series dataset suitable for visualization with Seaborn.

Outputs:
1. data/output/problem2_timeline.csv       -> time-series data per application
2. data/output/problem2_cluster_summary.csv -> aggregated cluster statistics
3. data/output/problem2_stats.txt           -> overall summary statistics
4. data/output/problem2_bar_chart.png       -> bar chart (applications per cluster)
5. data/output/problem2_density_plot.png    -> density plot (job durations)
"""

import s3fs, shutil
import glob
import os
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import (
    col, regexp_extract, input_file_name, try_to_timestamp, lit,
    min as spark_min, max as spark_max, count
)

# spark session (for both local test and cluster environment)
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
def run_problem2(input_path: str, output_dir: str, master_url: str = None, skip_spark: bool = False):

    # ensure output dir exists
    os.makedirs(output_dir, exist_ok=True)

    timeline_csv = os.path.join(output_dir, "problem2_timeline.csv")
    summary_csv = os.path.join(output_dir, "problem2_cluster_summary.csv")
    stats_txt = os.path.join(output_dir, "problem2_stats.txt")
    bar_chart = os.path.join(output_dir, "problem2_bar_chart.png")
    density_plot = os.path.join(output_dir, "problem2_density_plot.png")


    # data processing (unless --skip-spark)
    if not skip_spark:
        spark = get_spark("Problem2_ClusterUsageAnalysis", master_url)
        sc = spark.sparkContext
        sc._jsc.hadoopConfiguration().setBoolean(
            "mapreduce.input.fileinputformat.input.dir.recursive", True
        )

        print(f"Reading logs recursively from: {input_path}")
        logs_df = spark.read.text(f"{input_path.rstrip('/')}/*/**").withColumn("file_path", input_file_name())
        total_lines = logs_df.count()
        print(f"Total log lines read: {total_lines:,}")

        # parse timestamps and other fields
        logs_parsed = logs_df.select(
            regexp_extract('value', r'(\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})', 1).alias('timestamp'),
            regexp_extract("value", r"(INFO|WARN|ERROR|DEBUG)", 1).alias("log_level"),
            col("value").alias("message"),
            col("file_path")
        )

        # extract application_id and container_id from file path
        logs_parsed = logs_parsed.withColumn(
            "application_id", regexp_extract("file_path", r"application_(\d+_\d+)", 0)
        ).withColumn(
            "container_id", regexp_extract("file_path", r"container_(\d+_\d+_\d+_\d+)", 1)
        )

        # convert timestamp
        logs_parsed = logs_parsed.withColumn(
            "timestamp", try_to_timestamp(col("timestamp"), lit("yy/MM/dd HH:mm:ss"))
        )

        logs_parsed.cache()

        # compute start & end times per application
        app_times = logs_parsed.groupBy("application_id").agg(
            spark_min("timestamp").alias("start_time"),
            spark_max("timestamp").alias("end_time")
        )

        # extract cluster_id & app_number
        app_times = app_times.withColumn(
            "cluster_id", regexp_extract("application_id", r'application_(\d+)_\d+', 1)
        ).withColumn(
            "app_number", regexp_extract("application_id", r'_(\d+)$', 1)
        )

        # save timeline
        app_times.select("cluster_id", "application_id", "app_number", "start_time", "end_time") \
            .orderBy("cluster_id", "app_number") \
            .coalesce(1).write.csv(timeline_csv, header=True, mode="overwrite")
        print(f"Saved timeline CSV to {timeline_csv}")

        # compute cluster summary
        cluster_summary = app_times.groupBy("cluster_id").agg(
            count("application_id").alias("num_applications"),
            spark_min("start_time").alias("cluster_first_app"),
            spark_max("end_time").alias("cluster_last_app")
        )

        cluster_summary.coalesce(1).write.csv(summary_csv, header=True, mode="overwrite")
        print(f"Saved cluster summary CSV to {summary_csv}")

        # compute overall stats
        summary_pd = cluster_summary.toPandas()
        total_clusters = len(summary_pd)
        total_apps = summary_pd["num_applications"].sum()
        avg_apps = total_apps / total_clusters

        summary_lines = [
            f"Total unique clusters: {total_clusters}",
            f"Total applications: {total_apps}",
            f"Average applications per cluster: {avg_apps:.2f}\n",
            "Most heavily used clusters:",
        ]

        top = summary_pd.sort_values("num_applications", ascending=False).head(3)
        for _, r in top.iterrows():
            summary_lines.append(f"  Cluster {r['cluster_id']}: {r['num_applications']} applications")

        summary_df = spark.createDataFrame([Row(line=l) for l in summary_lines])
        summary_df.coalesce(1).write.mode("overwrite").text(stats_txt)
        print(f"Saved stats to {stats_txt}")

        spark.stop()
        print("Spark job completed successfully.")
    
    # visualization (use CSV outputs)

    print("Generating visualizations...")

    # help retrive data from s3
    def get_single_csv(path):
       
        # if s3 path
        if path.startswith("s3://") or path.startswith("s3a://"):
            fs = s3fs.S3FileSystem(anon=False)
            s3_path = path.replace("s3a://", "s3://").rstrip("/")
            files = fs.glob(f"{s3_path}/part-*.csv")
            if not files:
                raise FileNotFoundError(f"No CSV files found in {s3_path}")
            print(f"Found S3 file: {files[0]}")
            return f"s3://{files[0]}"

        # if local
        if os.path.isdir(path):
            csv_files = glob.glob(os.path.join(path, "*.csv"))
            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in {path}")
            print(f"Found local CSV: {csv_files[0]}")
            return csv_files[0]
        
        return path

    timeline_df = pd.read_csv(get_single_csv(timeline_csv))
    summary_df = pd.read_csv(get_single_csv(summary_csv))

    # help upload viz to s3
    def upload_to_s3(local_path, s3_path):
        fs = s3fs.S3FileSystem(anon=False)
        if s3_path.startswith("s3a://"):
            s3_path = s3_path.replace("s3a://", "s3://")
        with open(local_path, "rb") as f_in, fs.open(s3_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        print(f"Uploaded {local_path} → {s3_path}")

    # --- Bar chart ---
    plt.figure(figsize=(10, 6))
    sns.barplot(x="cluster_id", y="num_applications", data=summary_df, palette="crest")
    plt.title("Number of Applications per Cluster")
    plt.xlabel("Cluster ID")
    plt.ylabel("Number of Applications")
    for i, v in enumerate(summary_df["num_applications"]):
        plt.text(i, v + 1, str(v), ha="center", fontweight="bold")

    plt.savefig("problem2_bar_chart.png")
    upload_to_s3("problem2_bar_chart.png", bar_chart)

    plt.close()
    print(f"Saved {bar_chart}")

    # --- Density plot ---
    largest_cluster = summary_df.sort_values("num_applications", ascending=False).iloc[0]["cluster_id"]
    subset = timeline_df[timeline_df["cluster_id"].astype(str) == str(largest_cluster)].copy()
    subset["duration"] = pd.to_datetime(subset["end_time"]) - pd.to_datetime(subset["start_time"])
    subset["duration_min"] = subset["duration"].dt.total_seconds() / 60

    plt.figure(figsize=(10, 6))
    sns.histplot(subset["duration_min"], kde=True, log_scale=True)
    plt.title(f"Job Duration Distribution for Cluster {largest_cluster}\n n={len(subset)}")
    plt.xlabel("Duration (minutes, log scale)")
    plt.ylabel("Frequency")

    plt.savefig("problem2_density_plot.png")
    upload_to_s3("problem2_density_plot.png", density_plot)
    
    plt.close()
    print(f"Saved {density_plot}")

    print("Problem 2 completed successfully.")

# CLI Entrypoint
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Problem 2: Cluster Usage Analysis")
    parser.add_argument("--net-id", type=str, help="Your Georgetown NetID (e.g., rx67)")
    parser.add_argument("--input_path", type=str, default=None, help="S3 or local input path")
    parser.add_argument("--output_dir", type=str, default=None, help="S3 or local output directory")
    parser.add_argument("--master_url", type=str, default=None, help="Spark master URL")
    parser.add_argument("--skip-spark", action="store_true", help="Skip Spark and use existing CSVs")
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

    run_problem2(args.input_path, args.output_dir, args.master_url, args.skip_spark)
