"""
Utilities Module
Helper functions and utilities for the anomaly detection pipeline
"""

import os
import pandas as pd
import numpy as np


def validate_dataset(df, expected_features, target_column):
    """
    Validate that the dataset has the required structure
    
    Args:
        df: DataFrame to validate
        expected_features: List of expected feature column names
        target_column: Name of the target column
        
    Returns:
        tuple: (is_valid, missing_columns, extra_columns)
    """
    actual_columns = set(df.columns)
    expected_columns = set(expected_features + [target_column])
    
    missing_columns = expected_columns - actual_columns
    extra_columns = actual_columns - expected_columns
    
    is_valid = len(missing_columns) == 0
    
    return is_valid, list(missing_columns), list(extra_columns)


def check_data_quality(df):
    """
    Check data quality issues
    
    Args:
        df: DataFrame to check
        
    Returns:
        dict: Dictionary with quality metrics
    """
    quality_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict()
    }
    
    return quality_report


def print_quality_report(quality_report):
    """Print a formatted data quality report"""
    print("\n" + "=" * 80)
    print("DATA QUALITY REPORT")
    print("=" * 80)
    
    print(f"\nTotal Rows: {quality_report['total_rows']:,}")
    print(f"Total Columns: {quality_report['total_columns']}")
    print(f"Duplicate Rows: {quality_report['duplicate_rows']}")
    
    missing = quality_report['missing_values']
    if any(missing.values()):
        print("\nMissing Values:")
        for col, count in missing.items():
            if count > 0:
                pct = (count / quality_report['total_rows']) * 100
                print(f"  {col}: {count} ({pct:.2f}%)")
    else:
        print("\n✓ No missing values found")


def create_output_directory(output_dir):
    """
    Create output directory if it doesn't exist
    
    Args:
        output_dir: Path to output directory
        
    Returns:
        str: Absolute path to output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    return os.path.abspath(output_dir)


def save_model_summary(detector, output_dir):
    """
    Save a summary of model configurations and results
    
    Args:
        detector: NetworkAnomalyDetector instance
        output_dir: Directory to save the summary
    """
    summary = []
    summary.append("=" * 80)
    summary.append("MODEL SUMMARY")
    summary.append("=" * 80)
    
    if detector.supervised_model is not None:
        summary.append("\nSupervised Model (Random Forest):")
        summary.append(f"  Number of estimators: {detector.supervised_model.n_estimators}")
        summary.append(f"  Max depth: {detector.supervised_model.max_depth}")
        summary.append(f"  Class weight: {detector.supervised_model.class_weight}")
        
    if detector.unsupervised_model is not None:
        summary.append("\nUnsupervised Model (Isolation Forest):")
        summary.append(f"  Number of estimators: {detector.unsupervised_model.n_estimators}")
        summary.append(f"  Contamination: {detector.unsupervised_model.contamination}")
    
    summary.append("\n" + "=" * 80)
    
    # Save to file
    summary_path = os.path.join(output_dir, 'model_summary.txt')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary))
    
    print(f"\n✓ Model summary saved to '{summary_path}'")


def load_and_validate_data(data_path, expected_features, target_column):
    """
    Load data and perform validation
    
    Args:
        data_path: Path to CSV file
        expected_features: List of expected feature columns
        target_column: Name of target column
        
    Returns:
        DataFrame: Loaded and validated dataframe
    """
    # Load data
    df = pd.read_csv(data_path)
    
    # Validate structure
    is_valid, missing, extra = validate_dataset(df, expected_features, target_column)
    
    if not is_valid:
        print("\n⚠ WARNING: Dataset validation failed!")
        if missing:
            print(f"  Missing columns: {missing}")
        if extra:
            print(f"  Extra columns: {extra}")
    else:
        print("\n✓ Dataset validation passed")
    
    # Check quality
    quality_report = check_data_quality(df)
    print_quality_report(quality_report)
    
    return df


def get_class_distribution(y):
    """
    Get class distribution statistics
    
    Args:
        y: Array of labels
        
    Returns:
        dict: Distribution statistics
    """
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    
    distribution = {}
    for label, count in zip(unique, counts):
        distribution[int(label)] = {
            'count': int(count),
            'percentage': (count / total) * 100
        }
    
    return distribution


def print_class_distribution(y, label="Dataset"):
    """Print formatted class distribution"""
    dist = get_class_distribution(y)
    
    print(f"\n{label} Class Distribution:")
    for label, stats in dist.items():
        label_name = "Normal" if label == 0 else "Anomaly"
        print(f"  {label_name} (class {label}): {stats['count']:,} ({stats['percentage']:.2f}%)")
