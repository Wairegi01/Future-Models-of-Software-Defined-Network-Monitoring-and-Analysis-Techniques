# Dataset Path
DATA_PATH = r"C:\Users\iamwa\source\repos\Thesis\Network Analysis\Dataset\synthetic_network_traffic Re-Engineered.csv"

# Results Path
OUTPUT_DIR = r"C:\Users\iamwa\source\repos\Thesis\Network Analysis\Results"

# All Configurations

# Random Forest
RF_N_ESTIMATORS = 500
RF_MAX_DEPTH = 10
RF_MIN_SAMPLES_SPLIT = 10
RF_MIN_SAMPLES_LEAF = 5
RF_CLASS_WEIGHT = 'balanced'

# Isolation Forest
ISO_CONTAMINATION = 0.1
ISO_N_ESTIMATORS = 100
ISO_MAX_SAMPLES = 'auto'
ISO_MAX_FEATURES = 0.8

# Local Outlier Factor
LOF_N_NEIGHBORS = 35
LOF_CONTAMINATION = 0.1

# Data Split Configuration
TEST_SIZE = 0.3
RANDOM_STATE = 42

# Features that are right-skewed and benefit from log-transform
SKEWED_FEATURES = [
    'BytesSent',
    'BytesReceived',
    'PacketsSent',
    'PacketsReceived',
]

# Expected feature columns (including engineered features)
EXPECTED_FEATURES = [
    'SourceIP',
    'DestinationIP',
    'SourcePort',
    'DestinationPort',
    'Protocol',
    'BytesSent',
    'BytesReceived',
    'PacketsSent',
    'PacketsReceived',
    'Duration',
    'BytesPerPacketSent',
    'BytesPerPacketReceived',
]

# Target column name
TARGET_COLUMN = 'IsAnomaly'


# ===========================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================

FIGURE_DPI = 300               # Resolution for saved figures
FIGURE_FORMAT = 'png'          # Output format (png, pdf, svg)


# ============================================================================
# EXECUTION OPTIONS
# ============================================================================

RUN_SUPERVISED = True          # Run Random Forest
RUN_UNSUPERVISED = True        # Run Isolation Forest
RUN_TEMPORAL = True            # Run Temporal Analysis
RUN_BEHAVIORAL = True          # Run Behavioural Analysis
RUN_DEEP_LEARNING = True       # Run Autoencoder
VERBOSE = True                 # Print detailed output

# Autoencoder
AE_BOTTLENECK       = 4        # Bottleneck layer size
AE_ACTIVATION       = 'tanh'   # tanh outperforms relu for reconstruction
AE_LEARNING_RATE    = 5e-4
AE_MAX_ITER         = 500
AE_THRESHOLD_SIGMA  = 1.5      # Threshold = mean + 1.5 * std of normal errors

# Behavioural Analysis
BEHAVIOR_N_PROFILES = 5
BEHAVIOR_DBSCAN_MIN_SAMPLES = 100