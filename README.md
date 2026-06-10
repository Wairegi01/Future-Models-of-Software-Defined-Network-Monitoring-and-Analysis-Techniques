# SDN Anomaly Detection: A Multi-Modal Machine Learning Framework
A multi-modal machine learning framework for real-time anomaly detection in Software-Defined Networking (SDN) environments. Built as part of an MSc thesis in Telecommunications at Warsaw University of Technology.

## Overview
This project implements and evaluates five complementary machine learning techniques for detecting anomalous traffic in SDN environments. Rather than relying on a single model, the framework combines supervised, unsupervised, and deep learning approaches to achieve robust detection across different attack types and traffic patterns.
The system was evaluated on a 50,000-record synthetic SDN traffic dataset sampled from a 1-million-flow source, engineered to reflect realistic DDoS flood, port scan, and data exfiltration scenarios.

## Architecture

The framework consists of five components working together:

**1. Random Forest (Supervised)**
Trained on labelled traffic records for high-accuracy classification of known attack types. Evaluated on a balanced undersampled subset to avoid class imbalance bias.

**2. Isolation Forest (Unsupervised)**
Detects anomalies without labelled data by isolating outlier flows in feature space, which is useful for novel or previously unseen attack patterns.

**3. Autoencoder (Deep Learning)**
Learns a compressed representation of normal traffic. High reconstruction error signals anomalous behaviour. Trained on re-engineered data with clear class separation for reliable recall.

**4. Temporal Analysis with Rolling-Window Statistics**
Analyses traffic feature distributions over time using rolling windows. Near-zero autocorrelation (ρ < 0.015 across 200 lags) confirmed traffic is not sequentially dependent — providing empirical justification for choosing non-sequential ensemble models over LSTM-based approaches.

**5. K-Means Behavioural Profiling**
Clusters flow into behavioural profiles to identify structural deviations from normal traffic groups without requiring labels.

## Dataset

- **Source:** Synthetic SDN traffic dataset (1 million flows)
- **Working subset:** 50,000 records
- **Attack types modelled:** DDoS flood, port scan, data exfiltration
- **Preprocessing:** Balanced undersampling applied to address class imbalance before supervised training

## Project Structure

```
├── data/                   # Dataset and preprocessing scripts
├── models/
│   ├── random_forest.py
│   ├── isolation_forest.py
│   ├── autoencoder.py
│   ├── temporal_analysis.py
│   └── kmeans_profiling.py
├── results/                # Evaluation outputs and charts
└── README.md
```

## Academic Context

**Thesis title:** Future Models of Software-Defined Network Monitoring and Analysis Techniques

**Institution:** Warsaw University of Technology, Faculty of Electronics and Information Technology

**Programme:** MSc Telecommunications

**Supervisor:** Professor Kaleta

---

## License

This repository accompanies academic research. Please cite appropriately if you use or build on this work.

