"""
Advanced Analysis Runner
Run all future-focused SDN analyses
"""

from detector import NetworkAnomalyDetector
from temporal_analysis import run_temporal_analysis
from deep_learning import run_deep_learning_analysis
from behavior_analysis import run_behavior_analysis
from future_sdn import run_future_sdn_analysis
import config


def run_advanced_analyses(detector):
    """
    Run all advanced analyses for future SDN insights
    
    Args:
        detector: NetworkAnomalyDetector instance with trained models
    """
    print("\n" + "=" * 80)
    print("ADVANCED ANALYSES FOR FUTURE SDN")
    print("=" * 80)
    print("\nThis suite includes cutting-edge analyses that demonstrate")
    print("how ML/AI can revolutionize SDN monitoring and security.")
    
    results = {}
    
    # 1. Temporal Analysis
    print("\n" + "=" * 80)
    print("1/4: TEMPORAL PATTERN ANALYSIS")
    print("=" * 80)
    results['temporal'] = run_temporal_analysis(detector)
    
    # 2. Deep Learning Analysis
    print("\n" + "=" * 80)
    print("2/4: DEEP LEARNING ANALYSIS")
    print("=" * 80)
    results['deep_learning'] = run_deep_learning_analysis(detector)
    
    # 3. Behavioral Analysis
    print("\n" + "=" * 80)
    print("3/4: BEHAVIORAL ANALYSIS")
    print("=" * 80)
    results['behavior'] = run_behavior_analysis(detector)
    
    # 4. Future SDN Recommendations
    print("\n" + "=" * 80)
    print("4/4: FUTURE SDN RECOMMENDATIONS")
    print("=" * 80)
    results['future_sdn'] = run_future_sdn_analysis(detector)
    
    return results


def main():
    """Main function for advanced analysis"""
    
    print("\n" + "=" * 80)
    print("ADVANCED SDN MONITORING & ANALYTICS")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Data Path: {config.DATA_PATH}")
    print(f"  Output Directory: {config.OUTPUT_DIR}")
    
    # Initialize detector
    detector = NetworkAnomalyDetector(config.DATA_PATH, config.OUTPUT_DIR)
    
    # Run basic pipeline first
    print("\n" + "=" * 80)
    print("STEP 1: BASIC PIPELINE")
    print("=" * 80)
    detector.run_complete_pipeline(
        supervised=config.RUN_SUPERVISED,
        unsupervised=config.RUN_UNSUPERVISED
    )
    
    # Run advanced analyses
    print("\n" + "=" * 80)
    print("STEP 2: ADVANCED ANALYSES")
    print("=" * 80)
    advanced_results = run_advanced_analyses(detector)
    
    # Summary
    print("\n" + "=" * 80)
    print("ALL ANALYSES COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {config.OUTPUT_DIR}/")
    print("\nGenerated Visualizations:")
    print("  Basic Analysis:")
    print("    • eda_analysis.png")
    print("    • supervised_results.png")
    print("    • unsupervised_results.png")
    print("\n  Advanced Analysis:")
    print("    • temporal_analysis.png")
    print("    • traffic_forecasting.png")
    print("    • deep_learning_autoencoder.png")
    print("    • deep_learning_attention.png")
    print("    • deep_learning_ensemble.png")
    print("    • behavioral_profiling.png")
    print("    • density_clustering.png")
    print("    • flow_analysis.png")
    print("    • severity_scoring.png")
    print("    • future_sdn_vision.png")
    print("\nGenerated Reports:")
    print("    • supervised_predictions.csv")
    print("    • unsupervised_predictions.csv")
    print("    • future_sdn_report.md")
    
    print("\n" + "=" * 80)
    print("🎯 NEXT STEPS")
    print("=" * 80)
    print("\n1. Review future_sdn_report.md for detailed recommendations")
    print("2. Examine visualizations to understand patterns")
    print("3. Share findings with your network security team")
    print("4. Prioritize implementations based on your needs")
    print("\n✨ Your thesis now demonstrates state-of-the-art SDN analytics!")


if __name__ == "__main__":
    main()
