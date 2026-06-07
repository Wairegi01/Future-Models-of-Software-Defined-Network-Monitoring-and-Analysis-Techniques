"""
Export Module — saves predictions and result summaries to CSV.
"""

import pandas as pd
import numpy as np


def export_results(detector):
    print("\n" + "=" * 80)
    print("EXPORTING RESULTS")
    print("=" * 80)

    _export_supervised(detector)
    _export_unsupervised(detector)

    return detector


def _export_supervised(detector):
    if detector.supervised_model is None or detector.y_test is None:
        return

    y_proba = detector.supervised_model.predict_proba(detector.X_test)[:, 1]
    y_pred  = detector.supervised_model.predict(detector.X_test)

    df_out = pd.DataFrame({
        'True_Label':          detector.y_test,
        'Predicted_Label':     y_pred,
        'Anomaly_Probability': y_proba,
        'Correct':             detector.y_test == y_pred,
    })

    path = detector._get_output_path('supervised_predictions.csv')
    df_out.to_csv(path, index=False)
    print(f"  Supervised predictions saved to '{path}'")


def _export_unsupervised(detector):
    if detector.unsupervised_model is None:
        return

    # --- Training set ---
    train_scores = detector.unsupervised_model.score_samples(detector.X_train)
    train_binary = (detector.unsupervised_model.predict(detector.X_train) == -1).astype(int)

    train_out = pd.DataFrame({
        'Split':            'train',
        'Predicted_Anomaly': train_binary,
        'Anomaly_Score':     train_scores,
    })
    if detector.y_train is not None:
        train_out['True_Label'] = detector.y_train
        train_out['Correct']    = detector.y_train == train_binary

    # --- Test set (if available) ---
    if detector.X_test is not None:
        test_scores = detector.unsupervised_model.score_samples(detector.X_test)
        # Use the same contamination-based decision boundary for consistency
        test_preds  = detector.unsupervised_model.predict(detector.X_test)
        test_binary = (test_preds == -1).astype(int)

        test_out = pd.DataFrame({
            'Split':             'test',
            'Predicted_Anomaly': test_binary,
            'Anomaly_Score':     test_scores,
        })
        if detector.y_test is not None:
            test_out['True_Label'] = detector.y_test
            test_out['Correct']    = detector.y_test == test_binary

        combined = pd.concat([train_out, test_out], ignore_index=True)
    else:
        combined = train_out

    path = detector._get_output_path('unsupervised_predictions.csv')
    combined.to_csv(path, index=False)
    print(f"  Unsupervised predictions saved to '{path}'  "
          f"({len(combined):,} rows, train + test)")
