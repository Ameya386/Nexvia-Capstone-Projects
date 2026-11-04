"""Step 2 (Spark version) - same outputs as features.build_pandas(), computed with Apache Spark.

    python -m src.spark_features
Needs `pip install pyspark` and Java 8/11/17.
"""
from pyspark.sql import SparkSession, functions as F
from src import config as C


def main():
    spark = (SparkSession.builder.appName("rec-features").master("local[*]")
             .config("spark.driver.memory", "4g").getOrCreate())
    train = (spark.read.csv(str(C.PROC / "train.csv"), header=True, inferSchema=True)
             .select("user_idx", "item_idx", "rating")
             .withColumn("pos", (F.col("rating") >= C.POS_THRESHOLD).cast("int")))
    item = train.groupBy("item_idx").agg(F.count("*").alias("item_cnt"), F.avg("rating").alias("item_avg"),
                                         F.sum("pos").alias("item_pos"))
    user = train.groupBy("user_idx").agg(F.count("*").alias("user_cnt"), F.avg("rating").alias("user_avg"),
                                         F.sum("pos").alias("user_pos"))
    item.toPandas().to_csv(C.PROC / "item_features.csv", index=False)   # aggregated tables are small
    user.toPandas().to_csv(C.PROC / "user_features.csv", index=False)
    print("spark features done:", user.count(), "users,", item.count(), "items")
    spark.stop()


if __name__ == "__main__":
    main()
