# ============================================================
# Project : User Activity Pattern Detection using Apache Spark
# Dataset : E-Commerce RFM Customer Behavior Dataset
#           4,380 customers | 37 countries | 10 segments
# Tech    : Apache Spark (PySpark)
# ============================================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

spark = SparkSession.builder \
    .appName("UserActivityPatternDetection_RFM") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("=" * 60)
print("  Spark Session Initialized")
print(f"  Spark Version : {spark.version}")
print("=" * 60)


# -----------------------------------------------
# STEP 1: Load Dataset
# -----------------------------------------------
# We load the RFM CSV dataset into a Spark DataFrame.
# RFM = Recency, Frequency, Monetary
# These three values measure customer purchase behavior.

df = spark.read.csv(
    "ecom_data_rfm.csv",
    header=True,
    inferSchema=True
)

# Drop the unnamed index column
df = df.drop("_c0")

print("\n[STEP 1] Dataset Loaded")
print(f"  Total Records : {df.count()}")
print(f"  Total Columns : {len(df.columns)}")
df.printSchema()
df.show(5)


# -----------------------------------------------
# STEP 2: Data Preprocessing
# -----------------------------------------------

before = df.count()
df = df.filter(
    (F.col("Customer_Segment") != "NULL") &
    F.col("Customer_Segment").isNotNull()
)
after = df.count()

print(f"\n[STEP 2] Preprocessing")
print(f"  Rows before : {before}  |  After : {after}  |  Removed : {before - after}")

df = df.withColumn("Frequency", F.col("Frequency").cast("integer")) \
       .withColumn("Recency",   F.col("Recency").cast("integer"))   \
       .withColumn("Monetary",  F.col("Monetary").cast("double"))   \
       .withColumn("rankR",     F.col("rankR").cast("integer"))     \
       .withColumn("rankF",     F.col("rankF").cast("integer"))     \
       .withColumn("rankM",     F.col("rankM").cast("integer"))

df.cache()
df.show(3)


# -----------------------------------------------
# STEP 3: Most Common User Actions
#         = Customer Segment Distribution
# -----------------------------------------------

print("\n" + "=" * 60)
print("  SECTION A — EXPLORATORY ANALYSIS")
print("=" * 60)

print("\n[3A] Customer Segment Distribution:")
segment_counts = df.groupBy("Customer_Segment") \
    .agg(F.count("*").alias("customer_count")) \
    .orderBy(F.desc("customer_count"))

total = df.count()
segment_pct = segment_counts.withColumn(
    "percentage", F.round((F.col("customer_count") / total) * 100, 2)
)
segment_pct.show()


# -----------------------------------------------
# STEP 4: Activity Windows (replaces peak hour)
# -----------------------------------------------

print("[3B] Customer Activity Windows (Recency-based):")
df = df.withColumn(
    "activity_window",
    F.when(F.col("Recency") <= 30,  "Very Recent (0-30d)")
     .when(F.col("Recency") <= 90,  "Recent (31-90d)")
     .when(F.col("Recency") <= 180, "Moderate (91-180d)")
     .when(F.col("Recency") <= 270, "Inactive (181-270d)")
     .otherwise("Dormant (270d+)")
)

df.groupBy("activity_window") \
  .agg(F.count("*").alias("customers")) \
  .orderBy(F.desc("customers")).show()


# -----------------------------------------------
# STEP 5: Most Active Users
# -----------------------------------------------

print("[3C] Top 10 Most Frequent Buyers:")
df.select("CustomerID","Frequency","Recency","Monetary","Customer_Segment") \
  .orderBy(F.desc("Frequency")).show(10)

print("[3D] Top 10 Highest Spenders:")
df.select("CustomerID","Monetary","Frequency","Country","Customer_Segment") \
  .orderBy(F.desc("Monetary")).show(10)

print("[3E] Country-wise Distribution (Top 10):")
df.groupBy("Country") \
  .agg(
      F.count("*").alias("customers"),
      F.round(F.avg("Monetary"), 2).alias("avg_spend"),
      F.round(F.avg("Frequency"), 1).alias("avg_freq")
  ).orderBy(F.desc("customers")).show(10)


# -----------------------------------------------
# STEP 6: Pattern Detection
# -----------------------------------------------

print("\n" + "=" * 60)
print("  SECTION B — PATTERN DETECTION")
print("=" * 60)

print("\n[4A] RFM Score Group Patterns:")
df.groupBy("groupRFM") \
  .agg(
      F.count("*").alias("customers"),
      F.round(F.avg("Monetary"), 2).alias("avg_spend"),
      F.round(F.avg("Frequency"), 1).alias("avg_freq"),
      F.round(F.avg("Recency"), 1).alias("avg_recency")
  ).orderBy(F.desc("customers")).show(10)

print("[4B] Value Tier Analysis:")
df_tiered = df.withColumn(
    "value_tier",
    F.when(F.col("Monetary") >= 5000, "Premium")
     .when(F.col("Monetary") >= 1000, "High Value")
     .when(F.col("Monetary") >= 200,  "Mid Value")
     .otherwise("Low Value")
)
df_tiered.groupBy("value_tier") \
    .agg(
        F.count("*").alias("customers"),
        F.round(F.avg("Frequency"), 1).alias("avg_freq"),
        F.round(F.avg("Monetary"), 2).alias("avg_spend")
    ).orderBy(F.desc("customers")).show()

print("[4C] Loyalty Pattern Classification:")
df_loyalty = df.withColumn(
    "loyalty_pattern",
    F.when(F.col("Customer_Segment") == "Loyal Customers", "Loyal")
     .when(F.col("Customer_Segment").isin(
         "Potential Loyalist","New Customers","Promising"), "Growing")
     .when(F.col("Customer_Segment").isin(
         "At Risk","About To Sleep","Need Attention"), "At Risk")
     .otherwise("Lost")
)
df_loyalty.groupBy("loyalty_pattern") \
    .agg(
        F.count("*").alias("customers"),
        F.round(F.avg("Monetary"), 2).alias("avg_spend")
    ).orderBy(F.desc("customers")).show()


# -----------------------------------------------
# STEP 7: Anomaly Detection
# -----------------------------------------------

print("\n" + "=" * 60)
print("  SECTION C — ANOMALY DETECTION")
print("=" * 60)

stats = df.agg(
    F.mean("Frequency").alias("mf"), F.stddev("Frequency").alias("sf"),
    F.mean("Monetary").alias("mm"),  F.stddev("Monetary").alias("sm")
).collect()[0]

mean_f = stats["mf"]; std_f = stats["sf"]
mean_m = stats["mm"]; std_m = stats["sm"]
freq_threshold = mean_f + (2 * std_f)
low_mon_thresh = mean_m - std_m

print(f"\n  Frequency threshold (mean+2σ) : {round(freq_threshold, 1)}")
print(f"  Low spend threshold (mean-1σ) : {round(low_mon_thresh, 1)}")

df_anomaly = df.withColumn(
    "anomaly_flag",
    F.when(
        (F.col("Recency") > 300) & (F.col("Frequency") <= 2),
        "HIGH RISK - Churned"
    ).when(
        (F.col("Frequency") > freq_threshold) & (F.col("Monetary") < low_mon_thresh),
        "SUSPICIOUS - High Freq Low Spend"
    ).when(
        (F.col("Recency") > 180) & (F.col("Monetary") > 5000),
        "ALERT - High Value Going Inactive"
    ).when(
        F.col("Monetary") == 0,
        "ALERT - Zero Spend"
    ).otherwise("NORMAL")
)

print("\n[5A] Anomaly Summary:")
df_anomaly.groupBy("anomaly_flag") \
    .agg(F.count("*").alias("count")) \
    .orderBy(F.desc("count")).show()

flagged = df_anomaly.filter(F.col("anomaly_flag") != "NORMAL")
print(f"  Total Flagged : {flagged.count()} / {df.count()} ({round(flagged.count()/df.count()*100,1)}%)")

print("\n[5B] Sample Flagged Customers:")
flagged.select("CustomerID","Frequency","Recency","Monetary","Country","anomaly_flag") \
       .orderBy("anomaly_flag").show(15)


# -----------------------------------------------
# STEP 8: Final Summary
# -----------------------------------------------

print("\n" + "=" * 60)
print("  FINAL ANALYSIS SUMMARY")
print("=" * 60)

top_segment = segment_counts.first()["Customer_Segment"]
top_country = df.groupBy("Country").count().orderBy(F.desc("count")).first()["Country"]
loyal       = df.filter(F.col("Customer_Segment") == "Loyal Customers").count()
at_risk     = df.filter(F.col("Customer_Segment").isin("At Risk","About To Sleep")).count()
avg_spend   = df.agg(F.round(F.avg("Monetary"), 2)).collect()[0][0]

print(f"\n  Total Customers   : {df.count()}")
print(f"  Total Countries   : {df.select('Country').distinct().count()}")
print(f"  Top Segment       : {top_segment}")
print(f"  Top Country       : {top_country}")
print(f"  Loyal Customers   : {loyal}")
print(f"  At-Risk Customers : {at_risk}")
print(f"  Avg Spend         : £{avg_spend}")
print(f"  Anomaly Count     : {flagged.count()}")
print(f"  Anomaly Rate      : {round(flagged.count()/df.count()*100,1)}%")
print("\n  Project Execution Complete!")
print("=" * 60)

spark.stop()
