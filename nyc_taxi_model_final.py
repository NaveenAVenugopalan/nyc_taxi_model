# Databricks notebook source
# Download yellow taxi data 
def download_and_load_yellow_tripdata(year, month):
 
   import os
   import requests
   from pyspark.sql import SparkSession
   
   
   # Format month with leading zero if needed
   month_str = str(month).zfill(2)
   file_name = f"yellow_tripdata_{year}-{month_str}.parquet"
   url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{file_name}"
   
   # Define paths
   local_path = f"/tmp/{file_name}"
   dbfs_path = f"/FileStore/tables/{file_name}"
   
   # Download file if it doesn't exist locally . Download to local path as copy to dbfs was not working
   if os.path.exists(local_path):
       print(f"File {local_path} already exists. Skipping download.")
   else:
       print(f"Downloading {file_name}...")
       try:
           response = requests.get(url)
           if response.status_code == 200:
               with open(local_path, "wb") as f:
                   f.write(response.content)
               print(f"Download complete to {local_path}")
           else:
               print(f"Error: Unexpected status code {response.status_code}")
               return None
       except requests.exceptions.RequestException as e:
           print(f"An error occurred: {e}")
           return None
   
   # Copy file to DBFS
   dbutils.fs.cp(f"file:{local_path}", f"dbfs:{dbfs_path}")
   print(f"File copied to DBFS at {dbfs_path}")
   
   # Verify file exists in DBFS
   files = dbutils.fs.ls(f"/FileStore/tables/")
   file_exists = any(f.name == file_name for f in files)
   print(f"File verification in DBFS: {'Success' if file_exists else 'Failed'}")
   
   # Load the data from DBFS
   if file_exists:
       raw_df = spark.read.parquet(f"/FileStore/tables/{file_name}")
       print(f"Data loaded successfully. Row count: {raw_df.count()}")
       raw_df.printSchema()
       raw_df.show(5)
   else:
       print("ERROR: File not found in DBFS, cannot load data")
       return None
   
   print(f"Number of rows loaded to raw_df is: {raw_df.count()}")
   return raw_df

# COMMAND ----------

# Check and remove duplicate rows 

from pyspark.sql.functions import count
from pyspark.sql.window import Window

def process_duplicates(raw_df):
    
  
    # Count original records
    original_count = raw_df.count()
    print(f"Original record count before deduplication: {original_count}")
    
    # Count distinct records
    distinct_df = raw_df.dropDuplicates()
    distinct_count = distinct_df.count()
    print(f"Distinct record count after dedupliation: {distinct_count}")
    
    # Calculate duplicate count
    duplicate_count = original_count - distinct_count
    
      
    # Check if duplicates exist and handle accordingly
    if duplicate_count > 0:
        print(f"Duplicates found: {duplicate_count}")
        
        # Store duplicates in a separate dataframe
        window_spec = Window.partitionBy(*raw_df.columns)
        raw_df_with_count = raw_df.withColumn("occurrence_count", count().over(window_spec))
        raw_df_duplicates = raw_df_with_count.filter("occurrence_count > 1").drop("occurrence_count")
        
        # Use the distinct dataframe (duplicates removed)
        raw_dedup_df = distinct_df
        
        print(f"QUALITY REPORT: Number of duplicates removed: {duplicate_count}")
    else:
        print("No duplicates found")
        
        # No duplicate Keep original dataframe as is
        raw_dedup_df = raw_df
        raw_df_duplicates = None
                      
        print("QUALITY REPORT: No duplicates found")
    
    return raw_dedup_df, raw_df_duplicates

# COMMAND ----------


# Drop rows with NULL on selected columns 

from pyspark.sql.functions import col

def drop_null_rows(df):
    
    columns_to_check =  ["trip_distance","total_amount"]
    # Store the original row count
    original_count = df.count()
    
    # Create a dictionary to track how many nulls were found in each column
    null_counts = {}
    
    # Get counts of nulls for each column
    for column in columns_to_check:
        null_count = df.filter(col(column).isNull()).count()
        null_counts[column] = null_count
    
    # Drop rows with nulls in any of the specified columns
    clean_no_null_df = df
    for column in columns_to_check:
        clean_no_null_df = clean_no_null_df.filter(col(column).isNotNull())
    
    # Calculate the total rows removed
    final_count = clean_no_null_df.count()
    rows_removed = original_count - final_count
    
    print(f"Original row count: {original_count}")
    print(f"Final row count: {final_count}")
    print(f"Total rows removed: {rows_removed}")
    print("Null counts by column:")
    
    for column, count in null_counts.items():
        print(f"  - {column}: {count} rows")
    
    return clean_no_null_df

# COMMAND ----------

def convert_negative_to_positive_amounts(df):
    """
    Convert negative amounts to positive values for specified currency columns.
   
    """
   
    from pyspark.sql import functions as F
    
    # Hardcoded list of currency columns to fix
    columns_to_fix = [
        "fare_amount",
        "tip_amount",
        "tolls_amount",
        "extra",
        "mta_tax",
        "improvement_surcharge",
        "congestion_surcharge",
        "Airport_fee",
        "total_amount"
    ]
   
    # Original row count for reporting
    original_count = df.count()
   
    # Count negative values before conversion
    negative_counts = {}
    for col_name in columns_to_fix:
        try:
            neg_count = df.filter(F.col(col_name) < 0).count()
            negative_counts[col_name] = neg_count
        except:
            # Handle column not found
            negative_counts[col_name] = "Column not found"
   
    # Use Absolute value to convert negative to positive
    df_positive = df.select(
        *[F.abs(F.col(c)).alias(c) if c in columns_to_fix else F.col(c) for c in df.columns]
    )
   
    # Print summary information
    print("Negative currency values found (before conversion):")
    for col, count in negative_counts.items():
        print(f"  - {col}: {count}")
   
    print(f"\nTotal rows processed: {original_count}")
    print(f"All specified currency amounts converted to positive values.")
   
    return df_positive

# COMMAND ----------

# drop unwanted columns if needed , code not written as of now 



# COMMAND ----------

# Enrich data with some derived columns 

from pyspark.sql.functions import hour, to_date, date_format, when, col, round, unix_timestamp

def enhance_taxi_data(df):
   
   # Track the original row count
   original_count = df.count()
   
   # Apply all enhancements as a chain
   enhanced_df = (df
       # Pickup hour extracted from the tpep_pickup_datetime
       .withColumn(
           "pickup_hour",
           hour("tpep_pickup_datetime")
       )
       # Pickup date extracted from the tpep_pickup_datetime
       .withColumn(  
           "pickup_date",
           to_date("tpep_pickup_datetime")
       )
       # Pickup day of week extracted from the tpep_pickup_datetime
       .withColumn(  
           "pickup_day_of_week",
           date_format("tpep_pickup_datetime", "EEEE")
       )
       # Price per mile extracted from the trip_distance and fare_amount when not zero
       .withColumn(  
           "price_per_mile",
           when(col("trip_distance") > 0, round(col("fare_amount") / col("trip_distance"), 2)).otherwise(0)
       )
       # Trip duration in minutes calculated from pickup and dropoff timestamps
       .withColumn(
           "trip_duration_minutes",
           round((unix_timestamp("tpep_dropoff_datetime") - unix_timestamp("tpep_pickup_datetime")) / 60, 2)
       )
   )
   
   # Verify row count remains unchanged
   final_count = enhanced_df.count()
   
   # Display summary information
   print(f"Data enhancement complete:")
   print(f"- Original columns: {len(df.columns)}")
   print(f"- Enhanced columns: {len(enhanced_df.columns)}")
   print(f"- New columns added: {len(enhanced_df.columns) - len(df.columns)}")
   print(f"- Row count: {final_count} (unchanged from original: {original_count})")
   
   return enhanced_df

# COMMAND ----------

# : Create dimensional model that can be generated from original data set 

def create_dimensional_model(df):
    """
    Create a dimensional model from the cleaned data
    """
    print("Creating dimensional model...")
    from pyspark.sql import functions as sf
    # Dimension: Time
    dim_time = df.select(
        "tpep_pickup_datetime",
        "pickup_hour",
        "pickup_date",
        "pickup_day_of_week"
    ).distinct()
    
    
    # Nor creating a location dimension as https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv already has location_id and related details 

        
    # Fact Table: Trips
    fact_trips = df.select(
        sf.monotonically_increasing_id().alias("trip_id"),
        "VendorID",
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "PULocationID",
        "DOLocationID",
        "passenger_count",
        "payment_type",
        "RateCodeID",
        "trip_distance",
        "fare_amount",
        "tip_amount",
        "total_amount",
        "trip_duration_minutes",
        "price_per_mile"
    )
    
    # Display table counts
    print(f"Time dimension count: {dim_time.count()}")
    print(f"Fact table count: {fact_trips.count()}")
    dim_time.printSchema()
    fact_trips.printSchema()
    return {
        "dim_time": dim_time,
        "fact_trips": fact_trips
    }


# COMMAND ----------

# Save tables to Delta format in Databricks
def save_to_delta(tables_dict, base_path="/FileStore/tables/nyc_taxi"):
    """
    Save the dimensional model tables to Delta format
    """
    print(f"Saving tables to Delta format at {base_path}...")
    spark.sql(f"CREATE DATABASE IF NOT EXISTS nyc_db")
    print(f"Database nyc_db created or already exists")
    
    for table_name, df in tables_dict.items():
        table_path = f"{base_path}/{table_name}"
        print(f"Saving table {table_name} to {table_path}")
        df.printSchema()
        # Save as Delta table with Schema change adaption
        df.write.format("delta").option("overwriteSchema", "true").mode("overwrite").save(table_path)
        print(f"Saved {table_name} to Delta format")

        # Create temp view for SQL queries
        df.createOrReplaceTempView(table_name)
        # Register the table in the metastore
        spark.sql(f"CREATE TABLE IF NOT EXISTS nyc_db.{table_name} USING DELTA LOCATION '{table_path}'")
        print(f"Table nyc_db.{table_name} registered in metastore")
        
        
    
    print("All tables saved to Delta format successfully")

# COMMAND ----------

# create additional dimension not from main raw data 

def create_payment_type_dimension():
    
    from pyspark.sql.types import StructType, StructField, IntegerType, StringType
    
    # Create a DataFrame with payment type mappings
    payment_types_data = [
        (1, "Credit card"),
        (2, "Cash"),
        (3, "No charge"),
        (4, "Dispute"),
        (5, "Unknown"),
        (6, "Voided trip")
    ]
    
    schema = StructType([
        StructField("payment_type", IntegerType(), False),
        StructField("payment_type_desc", StringType(), False)
    ])
    dim_payment_type = spark.createDataFrame(payment_types_data, schema)
    delta_location = "dbfs:/FileStore/tables/dim_payment_type"
    dim_payment_type.write.format("delta").mode("overwrite").save(delta_location)

    spark.sql(f"CREATE TABLE IF NOT EXISTS nyc_db.dim_payment_type USING DELTA LOCATION '{delta_location}'")
    print(f"Table nyc_db.dim_payment_type registered in metastore")
    print("Sample data:")
    dim_payment_type.show(5)
    
    
    return dim_payment_type
    
    
def create_rate_code_dimension():
    
    from pyspark.sql.types import StructType, StructField, IntegerType, StringType
    
    # Create a DataFrame with rate code mappings
    rate_codes_data = [
        (1, "Standard rate"),
        (2, "JFK"),
        (3, "Newark"),
        (4, "Nassau or Westchester"),
        (5, "Negotiated fare"),
        (6, "Group ride"),
        (99, "Unknown")
    ]
    
    schema = StructType([
        StructField("RateCodeID", IntegerType(), False),
        StructField("rate_code_desc", StringType(), False)
    ])
    
    dim_rate_code=spark.createDataFrame(rate_codes_data, schema)
    delta_location = "dbfs:/FileStore/tables/dim_rate_code"
    dim_rate_code.write.format("delta").mode("overwrite").save(delta_location)

    spark.sql(f"CREATE TABLE IF NOT EXISTS nyc_db.dim_rate_code USING DELTA LOCATION '{delta_location}'")
    print(f"Table nyc_db.dim_rate_code registered in metastore")
    print("Sample data:")
    dim_rate_code.show(5)
    return dim_rate_code

   
   
    


# COMMAND ----------

# Dimension that needs to be downloaded 

import requests
def load_taxi_zone_dimension():    
    url = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
    local_path = "/tmp/taxi_zone_lookup.csv"

    print(f"Downloading taxi zone lookup from {url}...")
    response = requests.get(url)
    with open(local_path, "wb") as f:
        f.write(response.content)

    # Load to DBFS
    dbfs_path = "/FileStore/tables/taxi_zone_lookup.csv"
    dbutils.fs.cp(f"file:{local_path}", f"dbfs:{dbfs_path}")

    # Read the CSV into a DataFrame
    dim_taxi_zone = spark.read.option("header", "true").csv(dbfs_path)

    # Rename columns to match some naming convention 
    dim_taxi_zone = dim_taxi_zone.withColumnRenamed("LocationID", "location_id") \
                        .withColumnRenamed("Borough", "borough") \
                        .withColumnRenamed("Zone", "zone") \
                        .withColumnRenamed("service_zone", "service_zone")

    print(f"Loaded taxi zone dimension with {dim_taxi_zone.count()} rows")
    delta_location = "dbfs:/FileStore/tables/dim_taxi_zone_delta"
    dim_taxi_zone.write.format("delta").mode("overwrite").save(delta_location)

    spark.sql(f"CREATE TABLE IF NOT EXISTS nyc_db.dim_taxi_zone USING DELTA LOCATION '{delta_location}'")
    print(f"Table nyc_db.dim_taxi_zone registered in metastore")
    print("Sample data:")
    dim_taxi_zone.show(5)

    return dim_taxi_zone

# COMMAND ----------

# Summary of data quality after all transformations so far 

def perform_data_quality_check(raw_df, clean_df, tables_dict):
  
    print("\n---  DATA QUALITY ANALYSIS ---\n")
    
    # Calculate counts and percentages
    initial_count = raw_df.count()
    final_count = clean_df.count()
    retention_pct = (final_count / initial_count) * 100
    
    print(f"Initial row count: {initial_count}")
    print(f"Final row count after cleaning: {final_count}")
    print(f"Data retention: {retention_pct:.2f}%")
    # Generate warning if data quality issues exist
    if retention_pct < 85:
        print("\nWARNING: Significant data loss during cleaning (retention < 85%). Further investigation needed.")
    

    # Return a summary dictionary for reporting
    return {
        "initial_count": initial_count,
        "final_count": final_count,
        "retention_pct": retention_pct
        
    }


# COMMAND ----------

def create_zone_analytics_report():
    
    print("\n--- ZONE ANALYTICS REPORT ---\n")
    


    
    # Query 1: Top pickup boroughs by trip count
    print("Top pickup boroughs by trip count:")
    
    pickup_borough_query = """
        SELECT
            tz.borough as pickup_borough,
            COUNT(*) as trip_count,
            ROUND(AVG(f.total_amount), 2) as avg_fare,
            ROUND(SUM(f.total_amount), 2) as total_revenue
        FROM nyc_db.fact_trips f
        LEFT JOIN nyc_db.dim_taxi_zone tz ON f.PULocationID = tz.location_id
        WHERE tz.borough IS NOT NULL
        GROUP BY tz.borough
        ORDER BY trip_count DESC
    """
    spark.sql(pickup_borough_query).show()
    
    # Query 2: Top routes (zone pairs) by revenue
    print("Top 10 routes (zone pairs) by revenue:")
    
    top_routes_query = """
        SELECT
            pz.zone as pickup_zone,
            dz.zone as dropoff_zone,
            COUNT(*) as trip_count,
            ROUND(AVG(f.trip_distance), 2) as avg_distance,
            ROUND(AVG(f.trip_duration_minutes), 2) as avg_duration,
            ROUND(SUM(f.total_amount), 2) as total_revenue
        FROM nyc_db.fact_trips f
        LEFT JOIN nyc_db.dim_taxi_zone pz ON f.PULocationID = pz.location_id
        LEFT JOIN nyc_db.dim_taxi_zone dz ON f.DOLocationID = dz.location_id
        WHERE pz.zone IS NOT NULL AND dz.zone IS NOT NULL
        GROUP BY pz.zone, dz.zone
        ORDER BY total_revenue DESC
        LIMIT 10
    """
    spark.sql(top_routes_query).show()
    
    # Query 3: Average trip metrics by payment type
    print("Average trip metrics by payment type:")
    
    payment_metrics_query = """
        SELECT
            pt.payment_type_desc,
            COUNT(*) as trip_count,
            ROUND(AVG(f.trip_distance), 2) as avg_distance,
            ROUND(AVG(f.fare_amount), 2) as avg_fare,
            ROUND(AVG(f.tip_amount), 2) as avg_tip,
            ROUND(AVG(f.tip_amount / CASE WHEN f.fare_amount > 0 THEN f.fare_amount ELSE NULL END) * 100, 2) as tip_percentage
        FROM nyc_db.fact_trips f
        LEFT JOIN nyc_db.dim_taxi_zone pz ON f.PULocationID = pz.location_id
        LEFT JOIN nyc_db.dim_taxi_zone dz ON f.DOLocationID = dz.location_id
        LEFT JOIN nyc_db.dim_payment_type pt ON f.payment_type = pt.payment_type
        GROUP BY pt.payment_type_desc
        ORDER BY trip_count DESC
    """
    spark.sql(payment_metrics_query).show()
    
    # Query 4: Trip metrics by rate code
    print("Trip metrics by rate code:")
    
    rate_code_metrics_query = """
        SELECT
            rc.rate_code_desc,
            COUNT(*) as trip_count,
            ROUND(AVG(f.trip_distance), 2) as avg_distance,
            ROUND(AVG(f.trip_duration_minutes), 2) as avg_duration,
            ROUND(AVG(f.price_per_mile), 2) as avg_price_per_mile,
            ROUND(SUM(f.total_amount), 2) as total_revenue
        FROM nyc_db.fact_trips f
        LEFT JOIN nyc_db.dim_taxi_zone pz ON f.PULocationID = pz.location_id
        LEFT JOIN nyc_db.dim_taxi_zone dz ON f.DOLocationID = dz.location_id
        LEFT JOIN nyc_db.dim_rate_code rc ON f.RateCodeID = rc.RateCodeID
        GROUP BY rc.rate_code_desc
        ORDER BY trip_count DESC
    """
    spark.sql(rate_code_metrics_query).show()
    
    
    
    

# COMMAND ----------

def run_analytical_queries():
    
    print("\n--- ANALYTICS RESULTS ---\n")
    
    # Initialize results dictionary
    results = {}
    
    # Query 1: What are the busiest hours for taxi trips?
    print("Query 1: Busiest hours for taxi trips")
    busiest_hours = spark.sql("""
        SELECT 
            pickup_hour,
            COUNT(*) as trip_count
        FROM nyc_db.fact_trips f
        JOIN nyc_db.dim_time t ON f.tpep_pickup_datetime = t.tpep_pickup_datetime
        GROUP BY pickup_hour
        ORDER BY trip_count DESC
    """)
    busiest_hours.show(24)
    results["busiest_hours"] = busiest_hours
    
    # Query 2: Payment type analysis
    print("Payment type distribution analysis:")
    payment_analysis = spark.sql("""
        SELECT 
            pt.payment_type,pt.payment_type_desc,
            COUNT(*) as trip_count,
            ROUND(AVG(f.fare_amount), 2) as avg_fare,
            ROUND(AVG(f.tip_amount), 2) as avg_tip,
            ROUND(SUM(f.total_amount), 2) as total_revenue
        FROM nyc_db.fact_trips f
        JOIN nyc_db.dim_payment_type pt ON f.payment_type = pt.payment_type
        GROUP BY pt.payment_type,pt.payment_type_desc
        ORDER BY trip_count DESC
    """)
    payment_analysis.show()
    results["payment_analysis"] = payment_analysis
    
    # Query 3: Rate code analysis
    print("Rate code distribution analysis:")
    rate_code_analysis = spark.sql("""
        SELECT 
            rc.rate_code_desc,
            COUNT(*) as trip_count,
            ROUND(AVG(f.trip_distance), 2) as avg_distance,
            ROUND(AVG(f.total_amount), 2) as avg_fare,
            ROUND(SUM(f.total_amount), 2) as total_revenue
        FROM nyc_db.fact_trips f
        JOIN nyc_db.dim_rate_code rc ON f.RateCodeID = rc.RateCodeID
        GROUP BY rc.rate_code_desc
        ORDER BY trip_count DESC
    """)
    rate_code_analysis.show()
    results["rate_code_analysis"] = rate_code_analysis
    
    # Query 4: Cross-dimension analysis: Payment method by rate code
    print("Payment method by rate code:")
    cross_analysis = spark.sql("""
        SELECT 
            rc.rate_code_desc,
            pt.payment_type_desc,
            COUNT(*) as count
        FROM nyc_db.fact_trips f
        JOIN nyc_db.dim_payment_type pt ON f.payment_type = pt.payment_type
        JOIN nyc_db.dim_rate_code rc ON f.RateCodeID = rc.RateCodeID
        GROUP BY rc.rate_code_desc, pt.payment_type_desc
        ORDER BY rc.rate_code_desc, count DESC
    """)
    cross_analysis.show()
    results["cross_analysis"] = cross_analysis
    
    return results

# COMMAND ----------

def analyze_location_patterns():
    """
    Analyze trip patterns using the location dimension
    """
    print("\n--- LOCATION PATTERN ANALYSIS ---\n")
    

    
    # Query 1: Most popular pickup locations
    print("Top 10 most popular pickup locations:")
    popular_pickups = spark.sql("""
        SELECT 
            l.location_id,
            COUNT(*) as pickup_count,borough,zone,service_zone,
            AVG(f.trip_distance) as avg_distance,
            AVG(f.fare_amount) as avg_fare,
            SUM(f.total_amount) as total_revenue
        FROM nyc_db.fact_trips f
        JOIN nyc_db.dim_taxi_zone l ON f.PULocationID = l.location_id
        GROUP BY l.location_id,borough,zone,service_zone
        ORDER BY pickup_count DESC
        LIMIT 10
    """)
    popular_pickups.show()
    
    # Query 2: Most popular dropoff locations
    print("Top 10 most popular dropoff locations:")
    popular_dropoffs = spark.sql("""
        SELECT 
            l.location_id,borough,zone,service_zone,
            COUNT(*) as dropoff_count,
            AVG(f.trip_distance) as avg_distance,
            AVG(f.fare_amount) as avg_fare,
            SUM(f.total_amount) as total_revenue
        FROM nyc_db.fact_trips f
        JOIN nyc_db.dim_taxi_zone l ON f.DOLocationID = l.location_id
        GROUP BY l.location_id,borough,zone,service_zone
        ORDER BY dropoff_count DESC
        LIMIT 10
    """)
    popular_dropoffs.show()
    
    # Query 3: Top location pairs (routes)
    print("Top 10 most popular routes:")
    popular_routes = spark.sql("""
        SELECT 
            f.PULocationID as pickup_id,
            f.DOLocationID as dropoff_id,
            COUNT(*) as trip_count,
            AVG(f.trip_distance) as avg_distance,
            AVG(f.trip_duration_minutes) as avg_duration,
            SUM(f.total_amount) as total_revenue
        FROM nyc_db.fact_trips f
        GROUP BY f.PULocationID, f.DOLocationID
        ORDER BY trip_count DESC
        LIMIT 10
    """)
    popular_routes.show()
    
    
    return {
        "popular_pickups": popular_pickups,
        "popular_dropoffs": popular_dropoffs,
        "popular_routes": popular_routes
        
    }

# COMMAND ----------

def process_nyc_taxi_data(year, month):
    """
    End to End pipeline calling
    """
    print("Downloading uellow taxi data for Year {year} and Month {month}")
    raw_df= download_and_load_yellow_tripdata(year,month)

    print("Checking and removing duplicates ")     
    clean_df, raw_df_duplicates = process_duplicates(raw_df)
    
    print("Checking and removing nulls ")
    clean_no_null_df=drop_null_rows(clean_df)

    print("Converting negative amount to positive for currency columns")
    nonull_df_positive_amount = convert_negative_to_positive_amounts(clean_no_null_df)

    print("enriching the data frame")
    enhanced_df= enhance_taxi_data(nonull_df_positive_amount)
   
    print("Creating the dimensional tables from the downloaded and enhanced data")
    tables = create_dimensional_model(enhanced_df)
   
    print("Saving tables to Delta")
    save_to_delta(tables)
    
    print("Creating the additinal dimensional tables from lookup data")
    dim_payment_type = create_payment_type_dimension()
    dim_rate_code = create_rate_code_dimension()
    dim_taxi_zone = load_taxi_zone_dimension()
    
    # Add these to the tables dictionary
    tables["dim_payment_type"] = dim_payment_type
    tables["dim_rate_code"] = dim_rate_code
    tables["dim_taxi_zone"] = dim_taxi_zone
    
    # Perform data quality checks
    quality_report = perform_data_quality_check(raw_df, enhanced_df, tables)
    print("General analytical reports ")
    analysis_results = run_analytical_queries()
    
    # Print some reports     
    print("Zone related nalytical reports ")
    zone_analysis = create_zone_analytics_report()

    #print some pickup loation related rports 

    location_analysis=analyze_location_patterns()
    # Save tables to Delta format
    save_to_delta(tables, base_path="/FileStore/tables/nyc_taxi")

    # Verify the tables at the end
    verify_saved_tables()
    
    print("\nData processing complete.")
    return {
        "tables": tables,
        "quality_report": quality_report,
        "zone_analysis": zone_analysis
    }

# COMMAND ----------

def verify_saved_tables(limit=5):
    
    print("\n--- VERIFYING SAVED TABLES ---\n")
    
    # List of tables to verify
    tables_to_verify = [
        "dim_time", 
        "dim_taxi_zone", 
        "dim_payment_type",
        "dim_rate_code",
        "fact_trips" 
 
    ]
    
    # Set the database context
    spark.sql("USE nyc_db")
    
    # Check each table
    for table_name in tables_to_verify:
        try:
            print(f"\nSample data from {table_name}:")
            result = spark.sql(f"SELECT * FROM {table_name} LIMIT {limit}")
            result.show(truncate=False)
            row_count = spark.sql(f"SELECT COUNT(*) AS row_count FROM {table_name}").collect()[0][0]
            print(f"Total rows in {table_name}: {row_count:,}")
        except Exception as e:
            print(f"Error accessing table {table_name}: {str(e)}")
    

    
    print("\nTable verification complete.")



# COMMAND ----------

# call main function parameter year YYYY , Month M or MM
# Example process_nyc_taxi_data(2024,1)
# Example process_nyc_taxi_data(2024,12)
process_nyc_taxi_data(2021,8)

