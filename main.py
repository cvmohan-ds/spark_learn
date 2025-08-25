from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit, datediff, current_date, to_date
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
import pandas

def create_spark_session():
    """Initialize Spark session"""
    spark = SparkSession.builder \
        .appName("BODS_to_Pyspark") \
        .master("local[*]") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()
    return spark

def load_data(spark):
    """Load all required CSV files"""
    
    # Define schemas for better performance and data validation
    v_call_schema = StructType([
        StructField("License", StringType(), True),
        StructField("Caller", StringType(), True),
        StructField("call_type", StringType(), True),
        StructField("Call_info", StringType(), True),
        StructField("call_date", StringType(), True),
        StructField("time_spent_in_hours", IntegerType(), True)
    ])
    
    account_schema = StructType([
        StructField("Acc_no", StringType(), True),
        StructField("License", StringType(), True),
        StructField("vin_num", StringType(), True),
        StructField("cust_type", StringType(), True),
        StructField("date_started", StringType(), True)
    ])
    
    costing_schema = StructType([
        StructField("Cost_id", StringType(), True),
        StructField("cost_type", StringType(), True),
        StructField("work_rate_per_hour", DoubleType(), True),
        StructField("Work_labour_hours", IntegerType(), True)
    ])
    
    # Load CSV files
    v_call_df = spark.read.option("header", "true").schema(v_call_schema).csv("./data/v_call.csv")
    account_df = spark.read.option("header", "true").schema(account_schema).csv("./data/account.csv")
    costing_df = spark.read.option("header", "true").schema(costing_schema).csv("./data/costing.csv")
    
    return v_call_df, account_df, costing_df

def lookup_ext(input_df, account_df, spark):
    """
    Lookup function to join input_df with account information based on License
    """
    # Register dataframes as temporary views for SQL operations
    input_df.createOrReplaceTempView("input_calls")
    account_df.createOrReplaceTempView("accounts")
    
    # SQL query to join input with account data
    lookup_query = """
    SELECT 
        i.*,
        a.Acc_no,
        a.vin_num,
        a.cust_type,
        a.date_started
    FROM input_calls i
    LEFT JOIN accounts a ON i.License = a.License
    """
    
    transform_1_df = spark.sql(lookup_query)
    return transform_1_df

def apply_costing_lookup(transform_1_df, costing_df, spark):
    """
    Second transformation: Add costing information based on call_info
    """
    # Register dataframes as temporary views
    transform_1_df.createOrReplaceTempView("transform_1")
    costing_df.createOrReplaceTempView("costing")
    
    # SQL query to join with costing data
    costing_query = """
    SELECT 
        t1.*,
        c.work_rate_per_hour,
        c.Work_labour_hours
    FROM transform_1 t1
    LEFT JOIN costing c ON t1.Call_info = c.cost_type
    """
    
    transform_2_df = spark.sql(costing_query)
    return transform_2_df

def calculate_work_order_cost(transform_2_df):
    """
    Third transformation: Calculate work_order_cost_for_customer based on business rules
    """
    # Convert date_started to date type for date calculations
    transform_2_df = transform_2_df.withColumn(
        "date_started", 
        to_date(col("date_started"), "dd-MM-yyyy")
    )
    
    # Calculate days since date_started
    transform_2_df = transform_2_df.withColumn(
        "days_since_start",
        datediff(current_date(), col("date_started"))
    )
    
    # Apply business logic for cost calculation
    transform_3_df = transform_2_df.withColumn(
        "work_order_cost_for_customer",
        when(col("cust_type") == "Premium", 0.0)
        .when(col("cust_type") == "Regular", 
              col("work_rate_per_hour") * col("Work_labour_hours") * col("time_spent_in_hours"))
        .when(
            (col("cust_type") == "Free Tier") & (col("days_since_start") > 30),
            col("work_rate_per_hour") * col("Work_labour_hours") * col("time_spent_in_hours")
        )
        .when(
            (col("cust_type") == "Free Tier") & (col("days_since_start") <= 30),
            0.0
        )
        .otherwise(col("work_rate_per_hour") * col("Work_labour_hours") * col("time_spent_in_hours"))
    )

    transform_3_df = transform_3_df.withColumn(
        "is_free_tier_expired",
        when(
            (col("cust_type") == "Free Tier") & (col("days_since_start") > 30),
            "Yes"
            ).otherwise("No")
        
    )
    
    # Update customer type for free tire customers after 30 days
    transform_3_df = transform_3_df.withColumn(
        "cust_type",
        when(
            (col("cust_type") == "Free Tier") & (col("days_since_start") > 30),
            "Regular"
        ).otherwise(col("cust_type"))
    )
    
    return transform_3_df

def prepare_final_output(transform_3_df):
    """
    Prepare final output dataframe with required columns
    """
    final_df = transform_3_df.select(
        col("Acc_no").alias("acc_no"),
        col("License").alias("license"),
        col("vin_num").alias("vin_no"),
        col("cust_type"),
        col("is_free_tier_expired"),
        col("call_type"),
        col("Call_info").alias("call_info"),
        col("call_date"),
        col("work_rate_per_hour"),
        col("Work_labour_hours").alias("work_labor_rate"),  # Note: Following your output spec
        col("work_order_cost_for_customer").alias("final_cost")
    )
    
    return final_df

def main():
    """Main ETL pipeline execution"""
    # Initialize Spark session
    spark = create_spark_session()
    
    try:
        # Load data
        print("Loading data...")
        input_df, account_df, costing_df = load_data(spark)
        
        # Apply filter to get only 10 records as specified
        input_df = input_df.limit(15)
        
        print("Applying first transformation (account lookup)...")
        # First transformation: Account lookup
        transform_1_df = lookup_ext(input_df, account_df, spark)
        
        print("Applying second transformation (costing lookup)...")
        # Second transformation: Costing lookup
        transform_2_df = apply_costing_lookup(transform_1_df, costing_df, spark)
        
        print("Applying third transformation (cost calculation)...")
        # Third transformation: Cost calculation
        transform_3_df = calculate_work_order_cost(transform_2_df)
        
        print("Preparing final output...")
        # Prepare final output
        final_df = prepare_final_output(transform_3_df)
        
        # Show results (optional for debugging)
        print("Final results preview:")
        final_df.show(truncate=False)
        
        # Write to output CSV
        print("Writing output to WO_BILL.csv...")
        # final_df.coalesce(1) \
        #     .write \
        #     .mode("overwrite") \
        #     .option("header", "true") \
        #     .csv("./data/WO_BILL.csv")
        
        pandas_df = final_df.toPandas()
        pandas_df.to_csv("./data/WO_BILL.csv", index=False)
        print("ETL pipeline completed successfully!")
        
        # Print some statistics
        print(f"Total records processed: {final_df.count()}")
        print("Customer type distribution:")
        final_df.groupBy("cust_type").count().show()
        
    except Exception as e:
        print(f"Error in ETL pipeline: {str(e)}")
        raise
    
    finally:
        # Stop Spark session
        spark.stop()

if __name__ == "__main__":
    main()

# Additional utility functions for testing and validation

def create_sample_data(spark):
    """
    Create sample data for testing (optional)
    """
    from pyspark.sql import Row
    
    # Sample v_call data
    v_call_data = [
        Row(License="TS073373", Caller="Cust_1", call_type="SOS", 
            Call_info="Flat tyre", call_date="18-08-2024", time_spent_in_hours=2),
        Row(License="TS073374", Caller="Cust_2", call_type="SOS", 
            Call_info="Battery dead", call_date="19-08-2024", time_spent_in_hours=1)
    ]
    
    # Sample account data  
    account_data = [
        Row(Acc_no="A123456", License="TS073373", vin_num="VIN34445AA07", 
            cust_type="Premium", date_started="12-12-2023"),
        Row(Acc_no="A123457", License="TS073374", vin_num="VIN34445AA08", 
            cust_type="regular", date_started="15-01-2024")
    ]
    
    # Sample costing data
    costing_data = [
        Row(Cost_id="C_11", cost_type="Flat tyre", work_rate_per_hour=20.0, Work_labour_hours=1),
        Row(Cost_id="C_12", cost_type="Battery dead", work_rate_per_hour=15.0, Work_labour_hours=2)
    ]
    
    v_call_df = spark.createDataFrame(v_call_data)
    account_df = spark.createDataFrame(account_data)
    costing_df = spark.createDataFrame(costing_data)
    
    return v_call_df, account_df, costing_df

def run_with_sample_data():
    """
    Run the ETL pipeline with sample data for testing
    """
    spark = create_spark_session()
    
    try:
        print("Creating sample data...")
        input_df, account_df, costing_df = create_sample_data(spark)
        
        # Apply the same transformations
        input_df = input_df.limit(10)
        transform_1_df = lookup_ext(input_df, account_df, spark)
        transform_2_df = apply_costing_lookup(transform_1_df, costing_df, spark)
        transform_3_df = calculate_work_order_cost(transform_2_df)
        final_df = prepare_final_output(transform_3_df)
        
        print("Sample data processing results:")
        final_df.show(truncate=False)
        
    finally:
        spark.stop()

# Uncomment the line below to run with sample data for testing
# run_with_sample_data()