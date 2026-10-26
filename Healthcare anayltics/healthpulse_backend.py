"""
HealthPulse Analytics — PySpark backend
All data cleaning, aggregations, quality checks, model training and inference
run through Spark. Pandas conversion is only for the existing Streamlit/Plotly
presentation boundary.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional
import math
import os

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array

TARGET = "synthetic_cardiovascular_risk_label"
NUMERIC_FEATURES = [
    "age", "bmi", "glucose_mg_dl", "cholesterol_mg_dl",
    "systolic_bp", "diastolic_bp", "heart_rate"
]
RISK_LABELS = ["Low", "High"]


def get_spark(app_name: str = "HealthPulseAnalytics") -> SparkSession:
    """Create/reuse Spark. In Colab, call this after installing Java/PySpark."""
    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )


def load_dataset(spark: SparkSession, path: str) -> DataFrame:
    """Read CSV, normalize headers, cast clinical numeric fields, add patient_id."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = spark.read.option("header", True).option("inferSchema", True).option(
        "mode", "PERMISSIVE"
    ).csv(path)
    # Trim header names while preserving the dataset's field names.
    for old in df.columns:
        new = old.strip()
        if old != new:
            df = df.withColumnRenamed(old, new)
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df = df.withColumn(col, F.col(col).cast(DoubleType()))
    if "patient_id" not in df.columns:
        df = df.withColumn("patient_id", F.monotonically_increasing_id() + F.lit(1))
    if TARGET in df.columns:
        df = df.withColumn(
            TARGET,
            F.initcap(F.trim(F.col(TARGET).cast(StringType())))
        )
    return df


def to_pandas_for_ui(df: DataFrame, limit: Optional[int] = None):
    """Convert only at the UI boundary; use limit for large datasets."""
    if limit is not None:
        df = df.limit(int(limit))
    return df.toPandas()


def filter_dataset(
    df: DataFrame,
    genders: Optional[Iterable[str]] = None,
    age_range: Optional[tuple] = None,
    risks: Optional[Iterable[str]] = None,
) -> DataFrame:
    """Apply the dashboard's sidebar filters in Spark."""
    out = df
    if genders is not None and "gender" in out.columns:
        out = out.filter(F.col("gender").cast("string").isin([str(x) for x in genders]))
    if age_range is not None and "age" in out.columns:
        out = out.filter(F.col("age").between(float(age_range[0]), float(age_range[1])))
    if risks is not None and TARGET in out.columns:
        out = out.filter(F.col(TARGET).cast("string").isin([str(x) for x in risks]))
    return out


def _flag_expr(col_name: str):
    return F.lower(F.trim(F.col(col_name).cast("string"))).isin(
        "yes", "true", "1", "positive", "present"
    )


def overview_metrics(df: DataFrame) -> Dict[str, Any]:
    """Spark aggregations used by Executive Overview."""
    aggs = [F.count(F.lit(1)).alias("total")]
    for c in ["age", "bmi", "glucose_mg_dl", "systolic_bp", "cholesterol_mg_dl"]:
        if c in df.columns:
            aggs.append(F.avg(F.col(c)).alias("avg_" + c))
    row = df.agg(*aggs).first().asDict()
    total = int(row.get("total") or 0)
    high = low = 0
    if TARGET in df.columns:
        counts = {r[TARGET]: r["count"] for r in df.groupBy(TARGET).count().collect()}
        high = int(counts.get("High", 0) or 0)
        low = int(counts.get("Low", 0) or 0)
    diabetes = int(df.filter(_flag_expr("diabetes_indicator")).count()) if "diabetes_indicator" in df.columns else 0
    hypertension = int(df.filter(_flag_expr("hypertension")).count()) if "hypertension" in df.columns else 0
    return {
        "total": total,
        "avg_age": row.get("avg_age"),
        "avg_bmi": row.get("avg_bmi"),
        "avg_glucose": row.get("avg_glucose_mg_dl"),
        "avg_systolic_bp": row.get("avg_systolic_bp"),
        "avg_cholesterol": row.get("avg_cholesterol_mg_dl"),
        "high_count": high,
        "low_count": low,
        "high_pct": (high / total * 100) if total else 0.0,
        "diabetes_count": diabetes,
        "hypertension_count": hypertension,
    }


def data_quality_report(df: DataFrame) -> Dict[str, Any]:
    """Column-level missingness, uniqueness, duplicate count and schema."""
    rows = df.count()
    cols = []
    for field in df.schema.fields:
        c = field.name
        missing = df.filter(F.col(c).isNull() | (F.trim(F.col(c).cast("string")) == "")).count()
        unique = df.select(c).distinct().count()
        cols.append({
            "Column": c,
            "Data Type": field.dataType.simpleString(),
            "Missing Values": int(missing),
            "Missing %": round((missing / rows * 100), 2) if rows else 0.0,
            "Unique Values": int(unique),
        })
    duplicate_rows = rows - df.dropDuplicates().count()
    numeric_cols = [f.name for f in df.schema.fields if f.dataType.simpleString() in ("int", "bigint", "float", "double", "long", "decimal")]
    numeric_stats = df.select(*numeric_cols).summary().toPandas() if numeric_cols else None
    return {
        "rows": rows,
        "columns": len(df.columns),
        "missing_cells": sum(x["Missing Values"] for x in cols),
        "duplicate_rows": int(duplicate_rows),
        "column_quality": cols,
        "numeric_summary": numeric_stats,
        "schema": [(f.name, f.dataType.simpleString()) for f in df.schema.fields],
    }


def _prepare_training_data(df: DataFrame) -> DataFrame:
    missing = [c for c in NUMERIC_FEATURES if c not in df.columns]
    if missing:
        raise ValueError("Missing model features: " + ", ".join(missing))
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' is missing.")
    data = df.select(
        *[F.col(c).cast("double").alias(c) for c in NUMERIC_FEATURES],
        F.initcap(F.trim(F.col(TARGET).cast("string"))).alias(TARGET)
    ).filter(F.col(TARGET).isin(RISK_LABELS))
    return data


def train_model(df: DataFrame, model_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Train Spark MLlib RandomForest on an 80/20 stratified-ish split.
    Spark's randomSplit isn't stratified, so class balance is checked after split.
    """
    data = _prepare_training_data(df).cache()
    n = data.count()
    if n < 10:
        raise ValueError("At least 10 rows with High/Low target labels are required.")
    class_counts = {r[TARGET]: int(r["count"]) for r in data.groupBy(TARGET).count().collect()}
    if set(class_counts) != set(RISK_LABELS) or min(class_counts.values()) < 2:
        raise ValueError("Training data must contain at least two records for both High and Low labels.")

    # Fixed seed; random split is repeatable. Stratification is not directly supported by randomSplit.
    train_df, test_df = data.randomSplit([0.8, 0.2], seed=42)
    train_df = train_df.cache()
    test_df = test_df.cache()
    if train_df.count() == 0 or test_df.count() == 0:
        raise ValueError("Train/test split produced an empty partition; add more labeled rows.")

    # Encode labels with explicit stable mapping: Low=0, High=1.
    from pyspark.ml.feature import StringIndexer
    indexer = StringIndexer(inputCol=TARGET, outputCol="label", stringOrderType="alphabetAsc")
    imputer = Imputer(
        inputCols=NUMERIC_FEATURES,
        outputCols=[c + "__imputed" for c in NUMERIC_FEATURES],
        strategy="median"
    )
    assembler = VectorAssembler(
        inputCols=[c + "__imputed" for c in NUMERIC_FEATURES],
        outputCol="features",
        handleInvalid="keep"
    )
    rf = RandomForestClassifier(
        labelCol="label", featuresCol="features", predictionCol="prediction",
        probabilityCol="probability", numTrees=300, maxDepth=5, seed=42,
        featureSubsetStrategy="auto"
    )
    pipeline = Pipeline(stages=[indexer, imputer, assembler, rf])
    model = pipeline.fit(train_df)
    scored = model.transform(test_df).cache()

    # Spark evaluators operate on indexed labels.
    accuracy = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    ).evaluate(scored)
    # High-risk class is indexed as 0 because StringIndexer sorts labels alphabetically.
    high_tp = high_fp = high_fn = 0
    cm_tmp = scored.groupBy("label", "prediction").count().collect()
    for r in cm_tmp:
        actual, pred, count = int(r["label"]), int(r["prediction"]), int(r["count"])
        if actual == 0 and pred == 0:
            high_tp += count
        elif actual != 0 and pred == 0:
            high_fp += count
        elif actual == 0 and pred != 0:
            high_fn += count
    precision = high_tp / (high_tp + high_fp) if high_tp + high_fp else 0.0
    recall = high_tp / (high_tp + high_fn) if high_tp + high_fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0

    # Confusion matrix and per-class metrics, collected as small aggregate results.
    cm_rows = scored.groupBy("label", "prediction").count().collect()
    cm = [[0, 0], [0, 0]]
    for r in cm_rows:
        actual, pred = int(r["label"]), int(r["prediction"])
        if actual in (0, 1) and pred in (0, 1):
            cm[actual][pred] = int(r["count"])
    per_class = {}
    for idx, label in enumerate(["High", "Low"]):  # StringIndexer alphabetic: High=0, Low=1
        tp = cm[idx][idx]
        fp = sum(cm[a][idx] for a in range(2) if a != idx)
        fn = sum(cm[idx][p] for p in range(2) if p != idx)
        support = sum(cm[idx])
        per_class[label] = {
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1-score": (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0,
            "support": support,
        }

    # Feature importance from final RandomForest stage; feature order matches NUMERIC_FEATURES.
    rf_model = model.stages[-1]
    importance_rows = [
        {"Feature": name, "Importance": float(value)}
        for name, value in zip(NUMERIC_FEATURES, rf_model.featureImportances.toArray())
    ]
    importance_rows.sort(key=lambda x: x["Importance"], reverse=True)

    test_output = scored.select(
        *NUMERIC_FEATURES,
        F.col(TARGET).alias("Actual Risk"),
        F.when(F.col("prediction") == 0.0, F.lit("High")).otherwise(F.lit("Low")).alias("Predicted Risk")
    )
    test_pdf = test_output.toPandas()
    test_pdf["Correct"] = test_pdf["Actual Risk"] == test_pdf["Predicted Risk"]

    if model_dir:
        model.write().overwrite().save(model_dir)

    return {
        "model": model,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm,
        "classification_report": per_class,
        "importance": importance_rows,
        "train_rows": int(train_df.count()),
        "test_rows": int(test_df.count()),
        "total_rows": n,
        "class_counts": class_counts,
        "test_predictions": test_pdf,
        "error": None,
    }


def predict_one(model, measurements: Dict[str, Any]) -> Dict[str, Any]:
    """Run one patient's input through the trained Spark pipeline."""
    from pyspark.sql import Row
    missing = [c for c in NUMERIC_FEATURES if c not in measurements]
    if missing:
        raise ValueError("Missing prediction inputs: " + ", ".join(missing))
    values = {c: (float(measurements[c]) if measurements[c] is not None else None) for c in NUMERIC_FEATURES}
    spark = model.stages[0]._java_obj.sparkSession() if False else None
    # PipelineModel transforms a DataFrame from the active SparkSession.
    from pyspark.sql import SparkSession
    session = SparkSession.getActiveSession()
    if session is None:
        raise RuntimeError("No active SparkSession. Call get_spark() first.")
    scored = model.transform(session.createDataFrame([Row(**values)]))
    row = scored.select(TARGET if TARGET in scored.columns else F.lit(None).alias("unused")).first() if False else scored.select("prediction", "probability").first()
    # StringIndexer alphabetic order maps High=0, Low=1.
    predicted = "High" if int(row["prediction"]) == 0 else "Low"
    probs = row["probability"].toArray().tolist()
    return {
        "predicted_class": predicted,
        "probabilities": {"High": float(probs[0]), "Low": float(probs[1])},
        "confidence": float(probs[int(row["prediction"])]),
    }
