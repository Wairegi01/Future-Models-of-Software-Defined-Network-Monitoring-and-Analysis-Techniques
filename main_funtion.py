"""
Main Entry Point - Network Traffic Anomaly Detection
Run this file to execute the complete pipeline
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from detector import NetworkAnomalyDetector
import config


def main():
    """Main function to run the anomaly detection pipeline"""
    
    print("\n" + "=" * 80)
    print("NETWORK TRAFFIC ANOMALY DETECTION")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Data Path: {config.DATA_PATH}")
    print(f"  Output Directory: {config.OUTPUT_DIR}")
    print(f"  Supervised Learning: {'Enabled' if config.RUN_SUPERVISED else 'Disabled'}")
    print(f"  Unsupervised Learning: {'Enabled' if config.RUN_UNSUPERVISED else 'Disabled'}")
    print(f"  Temporal Analysis: {'Enabled' if config.RUN_TEMPORAL else 'Disabled'}")
    print(f"  Behavioural Analysis: {'Enabled' if config.RUN_BEHAVIORAL else 'Disabled'}")
    print(f"  Deep Learning:        {'Enabled' if config.RUN_DEEP_LEARNING else 'Disabled'}")

    # Initialize detector
    detector = NetworkAnomalyDetector(config.DATA_PATH, config.OUTPUT_DIR)

    # Run complete pipeline
    detector.run_complete_pipeline(
        supervised=config.RUN_SUPERVISED,
        unsupervised=config.RUN_UNSUPERVISED,
        temporal=config.RUN_TEMPORAL,
        behavioral=config.RUN_BEHAVIORAL,
        deep_learning=config.RUN_DEEP_LEARNING,
    )
    
    print("\n All analyses complete!")
    print(f"\nCheck the '{config.OUTPUT_DIR}' folder for all results.")


if __name__ == "__main__":
    main()
