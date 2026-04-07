"""
transform_star_schema.py
------------------------
Implements the Bronze → Silver → Gold medallion pipeline using PySpark for
the relational dimension of the lakehouse, targeting PostgreSQL.

Architecture justification:
  PySpark is used because the same transformation DAG scales from a single
  MacBook (local[*]) to a distributed cluster (Databricks / EMR / Dataproc)
  without code changes — satisfying the brief's scalability requirement.
  Spark's Catalyst optimiser minimises redundant data passes across the
  multi-source joins performed here.

Medallion layers:
  Bronze  → Raw data read from ingestion outputs, written to data/bronze/ as Parquet.
            Immutable. No transformations — byte-faithful copies of raw sources.
  Silver  → Cleaned, typed, deduplicated DataFrames in data/silver/ as Parquet.
            Nulls handled, dates cast, duplicates dropped, schema enforced.
  Gold    → Kimball star schema in data/gold/ as Parquet + loaded into PostgreSQL.
            Grain: one row per guest review in fact_review.

Star schema:
  fact_review          — core fact table (grain: per review)
  fact_review_weather  — bridge table: 10 daily weather rows per review
  dim_listing          — listing attributes
  dim_host             — host attributes
  dim_neighbourhood    — neighbourhood + borough + POI counts
  dim_date             — date spine with calendar attributes
  noise_borough_month  — pre-aggregated noise complaint counts by borough+month

Usage:
    python transform_star_schema.py

Environment variables (from .env):
    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    FloatType, DateType, BooleanType, LongType, DoubleType
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────

DATA_PIPELINE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR       = DATA_PIPELINE_DIR.parent
REPO_ROOT         = PROJECT_DIR.parent

RAW_AIRBNB_DIR    = DATA_PIPELINE_DIR / "raw" / "airbnb"
RAW_WEATHER_FILE  = DATA_PIPELINE_DIR / "raw" / "weather" / "nyc_weather.json"
RAW_NOISE_FILE    = DATA_PIPELINE_DIR / "raw" / "nyc_open_data" / "noise_complaints.json"
RAW_PLACES_DIR    = DATA_PIPELINE_DIR / "raw" / "places"

BRONZE_DIR        = PROJECT_DIR / "data" / "bronze"
SILVER_DIR        = PROJECT_DIR / "data" / "silver"
GOLD_DIR          = PROJECT_DIR / "data" / "gold"

LINEAGE_PATH      = DATA_PIPELINE_DIR / "processed" / "star_schema" / "lineage.json"


# ── Spark session ──────────────────────────────────────────────────────────────

def create_spark() -> SparkSession:
    """
    Create a local Spark session.
    local[*] uses all available cores — on a cluster, replace with the
    cluster master URL (e.g. spark://host:7077 or yarn).
    """
    return (
        SparkSession.builder
        .master("local[*]")
        .appName("NYC-Airbnb-StarSchema")
        # Increase driver memory for the 2.9M noise complaint records
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")   # reduce for local mode
        # Disable ANSI strict casting — source data contains empty strings in
        # numeric columns (e.g. bathrooms_text, review scores). With ANSI off,
        # invalid casts return NULL rather than raising CAST_INVALID_INPUT.
        .config("spark.sql.ansi.enabled", "false")
        .getOrCreate()
    )


# ── Snapshot detection ─────────────────────────────────────────────────────────

def latest_snapshot() -> Path:
    snapshots = sorted(RAW_AIRBNB_DIR.glob("up_to_*"))
    if not snapshots:
        raise FileNotFoundError(f"No Airbnb snapshots in {RAW_AIRBNB_DIR}")
    return snapshots[-1]


# ══════════════════════════════════════════════════════════════════════════════
# BRONZE LAYER — read raw sources into Spark DataFrames, persist as Parquet
# No transformations here — bronze is a faithful copy of the raw layer.
# ══════════════════════════════════════════════════════════════════════════════

def bronze_listings(spark: SparkSession, snapshot: Path) -> DataFrame:
    """
    Read the full listings.csv.gz (rich schema, ~36k rows) from the latest
    Airbnb snapshot. This contains listing attributes, host metadata, and
    pre-computed review scores — all needed for dimension tables.
    """
    path = str(snapshot / "listings.csv.gz")
    log.info(f"Bronze: reading listings from {path}")
    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "false")   # we cast types in silver
        .option("multiLine", "true")       # description fields contain newlines
        .option("escape", '"')
        .csv(path)
    )
    log.info(f"  → {df.count()} rows, {len(df.columns)} columns")
    _write_parquet(df, BRONZE_DIR / "listings")
    return df


def bronze_reviews(spark: SparkSession, snapshot: Path) -> DataFrame:
    """
    Read reviews.csv (listing_id, date) — the event log that drives fact_review.
    1M+ rows spanning 2009–2026.
    """
    path = str(snapshot / "reviews.csv")
    log.info(f"Bronze: reading reviews from {path}")
    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "false")
        .csv(path)
    )
    log.info(f"  → {df.count()} rows")
    _write_parquet(df, BRONZE_DIR / "reviews")
    return df


def bronze_weather(spark: SparkSession) -> DataFrame:
    """
    Read daily weather from nyc_weather.json. The Open-Meteo response stores
    all variables as parallel arrays keyed by date. We explode these into
    one row per day in the silver layer.
    """
    log.info(f"Bronze: reading weather from {RAW_WEATHER_FILE}")
    with open(RAW_WEATHER_FILE) as f:
        raw = json.load(f)

    daily = raw["daily"]
    # Zip parallel arrays into list of dicts — one dict per day
    records = [
        {
            "weather_date":       daily["time"][i],
            "temperature_max":    daily["temperature_2m_max"][i],
            "temperature_min":    daily["temperature_2m_min"][i],
            "precipitation_sum":  daily["precipitation_sum"][i],
            "windspeed_max":      daily["windspeed_10m_max"][i],
            "weathercode":        daily["weathercode"][i],
        }
        for i in range(len(daily["time"]))
    ]

    df = spark.createDataFrame(records)
    log.info(f"  → {df.count()} daily weather records")
    _write_parquet(df, BRONZE_DIR / "weather")
    return df


def bronze_noise(spark: SparkSession) -> DataFrame:
    """
    Read the 2.9M noise complaints from the Socrata JSON envelope.
    We load via Python json + Spark parallelise to handle the large nested file.
    """
    log.info(f"Bronze: reading noise complaints from {RAW_NOISE_FILE}")
    with open(RAW_NOISE_FILE) as f:
        raw = json.load(f)

    records = raw["records"]
    log.info(f"  Loaded {len(records)} complaint records into Python")

    df = spark.createDataFrame(records)
    log.info(f"  → {df.count()} rows in Spark DataFrame")
    _write_parquet(df, BRONZE_DIR / "noise_complaints")
    return df


def bronze_places(spark: SparkSession) -> DataFrame:
    """
    Read per-neighbourhood Places JSON files and union into one DataFrame.
    Used to enrich dim_neighbourhood with POI counts by type.
    """
    log.info(f"Bronze: reading Places data from {RAW_PLACES_DIR}")
    records = []
    for f in sorted(RAW_PLACES_DIR.glob("*.json")):
        if f.name == "lineage.json":
            continue
        with open(f) as fh:
            data = json.load(fh)
        nb = data.get("neighbourhood", "")
        for place_type, places in data.get("results_by_type", {}).items():
            for place in places:
                # Cast rating/lat/lng to float explicitly — Google Places API
                # returns integer ratings (e.g. 4) for some places and floats
                # (e.g. 4.5) for others. Mixed types cause CANNOT_MERGE_TYPE
                # during Spark schema inference.
                rating = place.get("rating")
                lat    = place.get("geometry", {}).get("location", {}).get("lat")
                lng    = place.get("geometry", {}).get("location", {}).get("lng")
                records.append({
                    "neighbourhood": nb,
                    "place_type":    place_type,
                    "place_id":      place.get("place_id"),
                    "name":          place.get("name"),
                    "rating":        float(rating) if rating is not None else None,
                    "lat":           float(lat)    if lat    is not None else None,
                    "lng":           float(lng)    if lng    is not None else None,
                })

    df = spark.createDataFrame(records)
    log.info(f"  → {df.count()} POI records across all neighbourhoods")
    _write_parquet(df, BRONZE_DIR / "places")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SILVER LAYER — clean, type-cast, deduplicate, validate
# ══════════════════════════════════════════════════════════════════════════════

def silver_listings(df: DataFrame) -> DataFrame:
    """
    Clean the raw listings DataFrame:
    - Select only the columns needed downstream
    - Cast numeric fields (Spark reads CSV as StringType by default)
    - Normalise boolean fields (t/f strings → boolean)
    - Deduplicate on listing id (keep latest scrape)
    """
    log.info("Silver: cleaning listings")
    df = (
        df
        .select(
            F.col("id").cast(LongType()).alias("listing_id"),
            # Host dimension fields
            F.col("host_id").cast(LongType()),
            F.col("host_name"),
            F.col("host_since").cast(DateType()),
            (F.col("host_is_superhost") == "t").alias("host_is_superhost"),
            F.col("host_response_time"),
            F.regexp_replace("host_response_rate", "%", "").cast(FloatType()).alias("host_response_rate"),
            F.regexp_replace("host_acceptance_rate", "%", "").cast(FloatType()).alias("host_acceptance_rate"),
            F.col("host_total_listings_count").cast(IntegerType()),
            (F.col("host_identity_verified") == "t").alias("host_identity_verified"),
            # Listing dimension fields
            F.col("neighbourhood_cleansed").alias("neighbourhood"),
            F.col("neighbourhood_group_cleansed").alias("borough"),
            F.col("latitude").cast(DoubleType()),
            F.col("longitude").cast(DoubleType()),
            F.col("room_type"),
            F.col("property_type"),
            F.col("accommodates").cast(IntegerType()),
            F.col("bedrooms").cast(IntegerType()),
            F.col("beds").cast(IntegerType()),
            # Parse bathrooms from text field e.g. "1.5 baths" → 1.5
            F.regexp_extract("bathrooms_text", r"([\d.]+)", 1).cast(FloatType()).alias("bathrooms"),
            F.col("minimum_nights").cast(IntegerType()),
            F.col("estimated_occupancy_l365d").cast(IntegerType()),
            (F.col("instant_bookable") == "t").alias("instant_bookable"),
            # Review scores (0–5 scale)
            F.col("review_scores_rating").cast(FloatType()),
            F.col("review_scores_accuracy").cast(FloatType()),
            F.col("review_scores_cleanliness").cast(FloatType()),
            F.col("review_scores_checkin").cast(FloatType()),
            F.col("review_scores_communication").cast(FloatType()),
            F.col("review_scores_location").cast(FloatType()),
            F.col("review_scores_value").cast(FloatType()),
            F.col("first_review").cast(DateType()),
            F.col("last_review").cast(DateType()),
            F.col("number_of_reviews").cast(IntegerType()),
        )
        # Drop listings with no listing_id or host_id — referential integrity
        .filter(F.col("listing_id").isNotNull() & F.col("host_id").isNotNull())
        # Deduplicate: if same listing appears in multiple snapshots, keep once
        .dropDuplicates(["listing_id"])
    )
    log.info(f"  → {df.count()} clean listings")
    _write_parquet(df, SILVER_DIR / "listings")
    return df


def silver_reviews(df: DataFrame) -> DataFrame:
    """
    Clean the raw reviews DataFrame:
    - Cast date column
    - Drop rows missing either field
    - Deduplicate on (listing_id, review_date) — same review may appear
      across multiple snapshot downloads
    """
    log.info("Silver: cleaning reviews")
    df = (
        df
        .withColumn("listing_id", F.col("listing_id").cast(LongType()))
        .withColumn("review_date", F.col("date").cast(DateType()))
        .drop("date")
        .filter(F.col("listing_id").isNotNull() & F.col("review_date").isNotNull())
        .dropDuplicates(["listing_id", "review_date"])
    )
    log.info(f"  → {df.count()} clean reviews")
    _write_parquet(df, SILVER_DIR / "reviews")
    return df


def silver_weather(df: DataFrame) -> DataFrame:
    """
    Clean the weather DataFrame:
    - Cast all numeric fields
    - Cast weather_date to DateType
    - No deduplication needed — Open-Meteo returns one row per date
    """
    log.info("Silver: cleaning weather")
    df = (
        df
        .withColumn("weather_date",      F.col("weather_date").cast(DateType()))
        .withColumn("temperature_max",   F.col("temperature_max").cast(FloatType()))
        .withColumn("temperature_min",   F.col("temperature_min").cast(FloatType()))
        .withColumn("precipitation_sum", F.col("precipitation_sum").cast(FloatType()))
        .withColumn("windspeed_max",     F.col("windspeed_max").cast(FloatType()))
        .withColumn("weathercode",       F.col("weathercode").cast(IntegerType()))
        .filter(F.col("weather_date").isNotNull())
    )
    log.info(f"  → {df.count()} clean daily weather records")
    _write_parquet(df, SILVER_DIR / "weather")
    return df


def silver_noise(df: DataFrame) -> DataFrame:
    """
    Clean the noise complaints DataFrame:
    - Parse created_date to DateType
    - Extract year, month for aggregation
    - Normalise borough field to match dim_neighbourhood.borough values
    - Drop records missing borough or date (unusable for the join)
    """
    log.info("Silver: cleaning noise complaints")

    # Borough name mapping: Socrata uses uppercase; normalise to title case
    # to match the neighbourhood_group_cleansed field in listings
    borough_map = {
        "MANHATTAN":     "Manhattan",
        "BROOKLYN":      "Brooklyn",
        "QUEENS":        "Queens",
        "BRONX":         "Bronx",
        "STATEN ISLAND": "Staten Island",
    }
    borough_expr = F.create_map(*[
        F.lit(k) if i % 2 == 0 else F.lit(v)
        for i, (k, v) in enumerate(
            (item for pair in borough_map.items() for item in [pair])
        )
    ])
    # Build the map expression properly
    map_expr = F.create_map(
        F.lit("MANHATTAN"),     F.lit("Manhattan"),
        F.lit("BROOKLYN"),      F.lit("Brooklyn"),
        F.lit("QUEENS"),        F.lit("Queens"),
        F.lit("BRONX"),         F.lit("Bronx"),
        F.lit("STATEN ISLAND"), F.lit("Staten Island"),
    )

    df = (
        df
        .withColumn("complaint_date",
                    F.to_date(F.col("created_date"), "yyyy-MM-dd'T'HH:mm:ss.SSS"))
        .withColumn("borough_clean", map_expr[F.upper(F.col("borough"))])
        .withColumn("year",  F.year("complaint_date"))
        .withColumn("month", F.month("complaint_date"))
        .filter(F.col("complaint_date").isNotNull() & F.col("borough_clean").isNotNull())
        .select(
            "unique_key",
            "complaint_date",
            "year",
            "month",
            F.col("borough_clean").alias("borough"),
            "complaint_type",
            "descriptor",
            F.col("latitude").cast(DoubleType()),
            F.col("longitude").cast(DoubleType()),
        )
    )
    log.info(f"  → {df.count()} clean noise complaint records")
    _write_parquet(df, SILVER_DIR / "noise_complaints")
    return df


def silver_places(df: DataFrame) -> DataFrame:
    """
    Clean the places DataFrame:
    - Cast rating and coordinates
    - Deduplicate on place_id per neighbourhood
    """
    log.info("Silver: cleaning places")
    df = (
        df
        .withColumn("rating", F.col("rating").cast(FloatType()))
        .withColumn("lat",    F.col("lat").cast(DoubleType()))
        .withColumn("lng",    F.col("lng").cast(DoubleType()))
        .filter(F.col("place_id").isNotNull())
        .dropDuplicates(["neighbourhood", "place_type", "place_id"])
    )
    log.info(f"  → {df.count()} clean POI records")
    _write_parquet(df, SILVER_DIR / "places")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# GOLD LAYER — Kimball star schema
# ══════════════════════════════════════════════════════════════════════════════

def build_dim_date(reviews_df: DataFrame) -> DataFrame:
    """
    Build the date dimension from all review dates.
    Season is defined meteorologically (Northern Hemisphere):
      Dec–Feb: Winter | Mar–May: Spring | Jun–Aug: Summer | Sep–Nov: Autumn
    """
    log.info("Gold: building dim_date")
    df = (
        reviews_df
        .select(F.col("review_date").alias("full_date"))
        .distinct()
        .withColumn("date_id",      F.date_format("full_date", "yyyyMMdd").cast(IntegerType()))
        .withColumn("year",         F.year("full_date"))
        .withColumn("month",        F.month("full_date"))
        .withColumn("month_name",   F.date_format("full_date", "MMMM"))
        .withColumn("quarter",      F.quarter("full_date"))
        .withColumn("day_of_week",  F.dayofweek("full_date"))   # 1=Sun … 7=Sat
        .withColumn("day_name",     F.date_format("full_date", "EEEE"))
        .withColumn("is_weekend",   F.dayofweek("full_date").isin([1, 7]))
        .withColumn("season",
            F.when(F.month("full_date").isin([12, 1, 2]), "Winter")
             .when(F.month("full_date").isin([3, 4, 5]),  "Spring")
             .when(F.month("full_date").isin([6, 7, 8]),  "Summer")
             .otherwise("Autumn")
        )
    )
    log.info(f"  → {df.count()} date records")
    _write_parquet(df, GOLD_DIR / "dim_date")
    return df


def build_dim_neighbourhood(listings_df: DataFrame, places_df: DataFrame) -> DataFrame:
    """
    Build the neighbourhood dimension, enriched with POI counts per category
    from Google Places data. POI counts give the agent a numerical signal for
    neighbourhood character (e.g. high restaurant count → entertainment area).
    """
    log.info("Gold: building dim_neighbourhood")

    # Base: distinct neighbourhoods from listings
    base = (
        listings_df
        .select("neighbourhood", "borough")
        .distinct()
        .filter(F.col("neighbourhood").isNotNull())
    )

    # POI counts per neighbourhood per type — pivot into columns
    poi_counts = (
        places_df
        .groupBy("neighbourhood", "place_type")
        .agg(F.countDistinct("place_id").alias("count"))
        .groupBy("neighbourhood")
        .pivot("place_type", ["restaurant", "bar", "lodging",
                               "tourist_attraction", "transit_station"])
        .agg(F.first("count"))
        # Rename to avoid spaces in column names
        .toDF("neighbourhood",
              "poi_restaurant", "poi_bar", "poi_lodging",
              "poi_tourist_attraction", "poi_transit_station")
    )

    df = (
        base
        .join(poi_counts, on="neighbourhood", how="left")
        # Replace nulls with 0 for neighbourhoods with no POIs of that type
        .fillna(0, subset=["poi_restaurant", "poi_bar", "poi_lodging",
                           "poi_tourist_attraction", "poi_transit_station"])
        .withColumn("poi_total",
            F.col("poi_restaurant") + F.col("poi_bar") + F.col("poi_lodging") +
            F.col("poi_tourist_attraction") + F.col("poi_transit_station")
        )
        # Surrogate key — hash of neighbourhood name for stability
        .withColumn("neighbourhood_id",
            F.abs(F.hash(F.col("neighbourhood"))).cast(IntegerType()))
    )
    log.info(f"  → {df.count()} neighbourhoods")
    _write_parquet(df, GOLD_DIR / "dim_neighbourhood")
    return df


def build_dim_host(listings_df: DataFrame) -> DataFrame:
    """
    Build the host dimension. One row per host_id — where a host has multiple
    listings we take the first record (host attributes are listing-independent).
    """
    log.info("Gold: building dim_host")
    df = (
        listings_df
        .select(
            "host_id", "host_name", "host_since",
            "host_is_superhost", "host_response_time",
            "host_response_rate", "host_acceptance_rate",
            "host_total_listings_count", "host_identity_verified",
        )
        .filter(F.col("host_id").isNotNull())
        # One row per host — deduplicate, keeping arbitrary record
        .dropDuplicates(["host_id"])
    )
    log.info(f"  → {df.count()} hosts")
    _write_parquet(df, GOLD_DIR / "dim_host")
    return df


def build_dim_listing(listings_df: DataFrame, dim_neighbourhood: DataFrame) -> DataFrame:
    """
    Build the listing dimension. Join with dim_neighbourhood to resolve the
    neighbourhood_id surrogate key — this is the foreign key used in fact_review.
    """
    log.info("Gold: building dim_listing")
    nb_keys = dim_neighbourhood.select("neighbourhood", "neighbourhood_id")

    df = (
        listings_df
        .join(nb_keys, on="neighbourhood", how="left")
        .select(
            "listing_id", "host_id", "neighbourhood_id",
            "room_type", "property_type", "accommodates",
            "bedrooms", "beds", "bathrooms", "minimum_nights",
            "estimated_occupancy_l365d", "instant_bookable",
            "review_scores_rating", "review_scores_accuracy",
            "review_scores_cleanliness", "review_scores_checkin",
            "review_scores_communication", "review_scores_location",
            "review_scores_value", "first_review", "last_review",
            "number_of_reviews", "latitude", "longitude",
        )
        .filter(F.col("listing_id").isNotNull())
    )
    log.info(f"  → {df.count()} listings")
    _write_parquet(df, GOLD_DIR / "dim_listing")
    return df


def build_fact_review(
    reviews_df: DataFrame,
    dim_listing: DataFrame,
    dim_neighbourhood: DataFrame,
    dim_date: DataFrame,
    noise_agg: DataFrame,
) -> DataFrame:
    """
    Build the core fact table at review grain.

    Joins:
      reviews → dim_listing       (listing_id → neighbourhood_id, host_id)
      reviews → dim_neighbourhood (neighbourhood_id → borough for noise join)
      reviews → dim_date          (review_date → date_id)
      reviews → noise_agg         (borough + year + month → complaint_count)

    Noise is joined at borough+month grain — the finest level achievable without
    a spatial join library. Complaint density (count per borough per month) is
    the analytically meaningful feature; individual records are not preserved.
    """
    log.info("Gold: building fact_review")

    date_keys = dim_date.select(
        F.col("full_date").alias("review_date"),
        "date_id"
    )

    # Borough lookup via dim_neighbourhood (borough is not in dim_listing)
    nb_borough = dim_neighbourhood.select("neighbourhood_id", "borough")

    df = (
        reviews_df
        .join(
            dim_listing.select("listing_id", "neighbourhood_id", "host_id"),
            on="listing_id",
            how="inner",
        )
        .join(date_keys, on="review_date", how="left")
        .join(nb_borough, on="neighbourhood_id", how="left")
        .withColumn("review_id",    F.monotonically_increasing_id())
        .withColumn("review_year",  F.year("review_date"))
        .withColumn("review_month", F.month("review_date"))
    )

    # Attach noise complaint count — join on borough + year + month
    df = (
        df
        .join(
            noise_agg.select("borough", "year", "month", "complaint_count"),
            on=[
                df["borough"]       == noise_agg["borough"],
                df["review_year"]   == noise_agg["year"],
                df["review_month"]  == noise_agg["month"],
            ],
            how="left",
        )
        .select(
            "review_id", "listing_id", "host_id",
            "neighbourhood_id", "date_id", "review_date",
            "complaint_count",
        )
    )

    log.info(f"  → {df.count()} review facts")
    _write_parquet(df, GOLD_DIR / "fact_review")
    return df


def build_noise_borough_month(noise_df: DataFrame) -> DataFrame:
    """
    Pre-aggregate noise complaints to borough-month grain.

    Justification: joining 2.9M individual complaint records to 1M reviews
    at review grain would be O(N×M) — computationally intractable. Aggregate
    density (complaint count per borough per month) is the analytically
    meaningful feature. This reduces the join to O(reviews × borough_months).

    Outputs:
      borough, year, month, complaint_count, top_complaint_types (array)
    """
    log.info("Gold: building noise_borough_month aggregation")
    df = (
        noise_df
        .groupBy("borough", "year", "month")
        .agg(
            F.count("unique_key").alias("complaint_count"),
            # Collect top complaint types for qualitative analysis
            F.collect_set("complaint_type").alias("complaint_types"),
        )
        .orderBy("borough", "year", "month")
    )
    log.info(f"  → {df.count()} borough-month noise records")
    _write_parquet(df, GOLD_DIR / "noise_borough_month")
    return df


def build_fact_review_weather(fact_review: DataFrame, weather_df: DataFrame) -> DataFrame:
    """
    Build the weather bridge table — the multi-valued relationship between
    a review and the 10 days of weather preceding it (estimated stay window).

    Pattern: Kimball bridge table (The Data Warehouse Toolkit, Ch. 6).
    Grain: one row per (review_id, weather_date) pair.
    Each review maps to up to 10 weather rows: review_date − 10 to review_date − 1.

    The daily granularity is deliberately preserved here — downstream analysts
    can choose their own aggregation (mean temp, count of rainy days, etc.)
    rather than having an aggregation baked in at transformation time.

    Implementation note: we use a cross-join filtered to the 10-day window.
    This is safe because weather_df is small (6,119 rows). The join produces
    at most fact_review.count × 10 rows.
    """
    log.info("Gold: building fact_review_weather bridge table")

    # Add days_before_review column to weather for interpretability
    # We join where: review_date - 10 <= weather_date < review_date
    df = (
        fact_review
        .select("review_id", "review_date")
        # Broadcast hint: weather_df is 6k rows — forces Spark to replicate it
        # to every executor rather than shuffling fact_review (1M rows).
        # Without this hint Spark may choose a sort-merge join which is ~10x slower.
        .crossJoin(F.broadcast(weather_df))
        # Filter to 10-day window before review date
        .filter(
            (F.col("weather_date") >= F.date_sub(F.col("review_date"), 10)) &
            (F.col("weather_date") <  F.col("review_date"))
        )
        .withColumn(
            "days_before_review",
            F.datediff(F.col("review_date"), F.col("weather_date"))
        )
        .select(
            "review_id", "weather_date", "days_before_review",
            "temperature_max", "temperature_min",
            "precipitation_sum", "windspeed_max", "weathercode",
        )
    )
    log.info(f"  → {df.count()} review-weather bridge rows")
    _write_parquet(df, GOLD_DIR / "fact_review_weather")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# POSTGRES LOAD — write gold tables to PostgreSQL via pandas + psycopg2
# Using pandas.to_sql rather than Spark JDBC to avoid JAR dependency.
# In production, replace with spark.write.jdbc() + the PostgreSQL JDBC driver.
# ══════════════════════════════════════════════════════════════════════════════

def get_postgres_engine():
    """Build a SQLAlchemy engine from .env credentials."""
    try:
        from sqlalchemy import create_engine
        host = os.getenv("POSTGRES_HOST", "localhost")
        port = os.getenv("POSTGRES_PORT", "5432")
        db   = os.getenv("POSTGRES_DB",   "airbnb")
        user = os.getenv("POSTGRES_USER", "postgres")
        pw   = os.getenv("POSTGRES_PASSWORD", "postgres")
        url  = f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{db}"
        return create_engine(url)
    except Exception as e:
        log.warning(f"Could not create PostgreSQL engine: {e}")
        return None


def write_to_postgres(df: DataFrame, table_name: str, engine) -> bool:
    """
    Convert a Spark DataFrame to pandas and write to PostgreSQL.
    if_exists='replace' drops and recreates the table on each run — idempotent.
    For append-only production pipelines, use 'append' with deduplication logic.
    """
    if engine is None:
        log.warning(f"  Skipping PostgreSQL write for {table_name} — no engine")
        return False
    try:
        log.info(f"  Writing {table_name} to PostgreSQL")
        pdf = df.toPandas()
        pdf.to_sql(table_name, engine, if_exists="replace", index=False,
                   method="multi", chunksize=10_000)
        log.info(f"  → {len(pdf)} rows written to {table_name}")
        return True
    except Exception as e:
        log.error(f"  Failed to write {table_name}: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _write_parquet(df: DataFrame, path: Path):
    """Write a DataFrame to Parquet, overwriting any existing data."""
    path.mkdir(parents=True, exist_ok=True)
    df.write.mode("overwrite").parquet(str(path))
    log.info(f"  Parquet written to {path}")


def write_lineage(lineage: dict):
    LINEAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LINEAGE_PATH, "w") as f:
        json.dump(lineage, f, indent=2)
    log.info(f"Lineage written to {LINEAGE_PATH}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run():
    load_dotenv(REPO_ROOT / ".env")

    spark   = create_spark()
    engine  = get_postgres_engine()
    snapshot = latest_snapshot()
    run_start = datetime.now(timezone.utc).isoformat()

    lineage = {"run_started_at": run_start, "tables": {}}

    # ── Bronze ──────────────────────────────────────────────────────────────
    log.info("═══ BRONZE LAYER ═══")
    b_listings = bronze_listings(spark, snapshot)
    b_reviews  = bronze_reviews(spark, snapshot)
    b_weather  = bronze_weather(spark)
    b_noise    = bronze_noise(spark)
    b_places   = bronze_places(spark)

    # ── Silver ──────────────────────────────────────────────────────────────
    log.info("═══ SILVER LAYER ═══")
    s_listings = silver_listings(b_listings)
    s_reviews  = silver_reviews(b_reviews)
    s_weather  = silver_weather(b_weather)
    s_noise    = silver_noise(b_noise)
    s_places   = silver_places(b_places)

    # ── Gold: Dimensions ────────────────────────────────────────────────────
    log.info("═══ GOLD LAYER — DIMENSIONS ═══")
    dim_date          = build_dim_date(s_reviews)
    dim_neighbourhood = build_dim_neighbourhood(s_listings, s_places)
    dim_host          = build_dim_host(s_listings)
    dim_listing       = build_dim_listing(s_listings, dim_neighbourhood)
    noise_month       = build_noise_borough_month(s_noise)

    # ── Gold: Facts ─────────────────────────────────────────────────────────
    log.info("═══ GOLD LAYER — FACTS ═══")
    fact_review         = build_fact_review(s_reviews, dim_listing, dim_neighbourhood, dim_date, noise_month)
    fact_review_weather = build_fact_review_weather(fact_review, s_weather)

    # ── PostgreSQL Load ─────────────────────────────────────────────────────
    log.info("═══ POSTGRESQL LOAD ═══")
    tables = {
        "dim_date":             dim_date,
        "dim_neighbourhood":    dim_neighbourhood,
        "dim_host":             dim_host,
        "dim_listing":          dim_listing,
        "noise_borough_month":  noise_month,
        "fact_review":          fact_review,
        "fact_review_weather":  fact_review_weather,
    }
    for name, df in tables.items():
        ok = write_to_postgres(df, name, engine)
        lineage["tables"][name] = {
            "rows":  df.count(),
            "pg_ok": ok,
        }

    lineage["run_finished_at"] = datetime.now(timezone.utc).isoformat()
    write_lineage(lineage)
    spark.stop()
    log.info("Star schema transformation complete.")


if __name__ == "__main__":
    run()
