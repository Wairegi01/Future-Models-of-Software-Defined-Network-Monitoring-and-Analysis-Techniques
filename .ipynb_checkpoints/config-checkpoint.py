# Path to my Dataset
DATA_PATH = r"C:\Users\iamwa\source\repos\Thesis\Network Analysis\Dataset\synthetic_network_traffic.csv"

# Path to save my results
OUTPUT_DIR = r"C:\Users\iamwa\source\repos\Thesis\Network Analysis\Results"


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MAX_SAMPLES = 250000  # Number of samples to use

# Random Forest (Supervised Learning)
RF_N_ESTIMATORS = 100          # Number of trees
RF_MAX_DEPTH = 10              # Maximum tree depth
RF_MIN_SAMPLES_SPLIT = 5       # Minimum samples to split a node
RF_MIN_SAMPLES_LEAF = 2        # Minimum samples in a leaf
RF_CLASS_WEIGHT = 'balanced'   # Handle class imbalance

# Isolation Forest (Unsupervised Learning)
ISO_CONTAMINATION = 0.1        # Expected proportion of anomalies (10%)
ISO_N_ESTIMATORS = 100         # Number of trees
ISO_MAX_SAMPLES = 'auto'       # Samples per tree


# ============================================================================
# DATA SPLIT CONFIGURATION
# ============================================================================

TEST_SIZE = 0.2                # 20% for testing, 80% for training
RANDOM_STATE = 42              # For reproducibility


# Expected feature columns in the dataset
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
    'Duration'
]

# Target column name
TARGET_COLUMN = 'IsAnomaly'


# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================

FIGURE_DPI = 300               # Resolution for saved figures
FIGURE_FORMAT = 'png'          # Output format (png, pdf, svg)


# ============================================================================
# EXECUTION OPTIONS
# ============================================================================

RUN_SUPERVISED = True          # Run Random Forest
RUN_UNSUPERVISED = True        # Run Isolation Forest
VERBOSE = True                 # Print detailed output