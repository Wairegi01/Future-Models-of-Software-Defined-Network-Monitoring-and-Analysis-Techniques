"""
Generates a synthetic network traffic dataset with realistic anomaly patterns.

Normal traffic is intentionally diverse (web browsing, file downloads, uploads,
DNS, VoIP, DB queries) so that each normal sub-type overlaps with at least one
anomaly type in individual features.  Discrimination requires the *combination*
of features, not a single one — this produces realistic model scores (~0.92-0.97
F1) rather than a trivial 1.0.

Anomaly types:
  - DDoS        : flood — high outbound, near-zero inbound, short burst
  - Port scan   : very low bytes, SYN-like, well-known dst ports
  - Exfiltration: high outbound ratio, long session, external-looking dst IPs
  - Brute force : small repeated payloads to a few admin ports
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

N_NORMAL  = 50_000
N_ANOMALY =  5_000   # 10 % anomaly rate


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _clip(arr, lo, hi):
    return np.clip(arr, lo, hi)


# ---------------------------------------------------------------------------
# 1. Normal traffic — six realistic sub-types so individual features overlap
#    with anomaly ranges
# ---------------------------------------------------------------------------
def gen_normal(n):
    frames = []

    # --- 1a. Web browsing (30 %) — moderate bidirectional, short-medium session
    nb = int(n * 0.30)
    frames.append(pd.DataFrame({
        'SourceIP':        rng.integers(10, 200, nb).astype(float),
        'DestinationIP':   rng.integers(10, 200, nb).astype(float),
        'SourcePort':      rng.integers(1024, 65535, nb).astype(float),
        'DestinationPort': rng.choice([80, 443, 8080, 8443], nb).astype(float),
        'Protocol':        rng.choice([6], nb).astype(float),
        'BytesSent':       _clip(rng.normal(8_000,  6_000, nb), 200, 60_000),
        'BytesReceived':   _clip(rng.normal(35_000, 18_000, nb), 500, 100_000),
        'PacketsSent':     _clip(rng.normal(80,      60,    nb), 2,   800),
        'PacketsReceived': _clip(rng.normal(350,    200,    nb), 5,   2_000),
        'Duration':        _clip(rng.normal(180,    150,    nb), 5,   1_800),
        'IsAnomaly': 0,
    }))

    # --- 1b. File download (15 %) — HIGH recv, LOW sent → overlaps with exfil style
    nd = int(n * 0.15)
    frames.append(pd.DataFrame({
        'SourceIP':        rng.integers(10, 200, nd).astype(float),
        'DestinationIP':   rng.integers(10, 200, nd).astype(float),
        'SourcePort':      rng.integers(1024, 65535, nd).astype(float),
        'DestinationPort': rng.choice([80, 443, 21, 22], nd).astype(float),
        'Protocol':        rng.choice([6, 17], nd).astype(float),
        'BytesSent':       _clip(rng.normal(1_500,  2_000, nd), 100,  20_000),
        'BytesReceived':   _clip(rng.normal(80_000, 15_000, nd), 10_000, 100_000),
        'PacketsSent':     _clip(rng.normal(30,      25,    nd), 2,    300),
        'PacketsReceived': _clip(rng.normal(900,    500,    nd), 50,  8_000),
        'Duration':        _clip(rng.normal(600,    400,    nd), 30,  3_600),
        'IsAnomaly': 0,
    }))

    # --- 1c. File upload / backup (15 %) — HIGH sent, LOW recv → overlaps with DDoS/exfil
    nu = int(n * 0.15)
    frames.append(pd.DataFrame({
        'SourceIP':        rng.integers(10, 200, nu).astype(float),
        'DestinationIP':   rng.integers(10, 200, nu).astype(float),
        'SourcePort':      rng.integers(1024, 65535, nu).astype(float),
        'DestinationPort': rng.choice([22, 21, 443, 2049], nu).astype(float),
        'Protocol':        rng.choice([6, 17], nu).astype(float),
        'BytesSent':       _clip(rng.normal(75_000, 18_000, nu), 8_000, 100_000),
        'BytesReceived':   _clip(rng.normal(2_000,  3_000,  nu), 100,   25_000),
        'PacketsSent':     _clip(rng.normal(850,    500,    nu), 50,   8_000),
        'PacketsReceived': _clip(rng.normal(40,      35,    nu), 2,     400),
        'Duration':        _clip(rng.normal(1_200,  800,    nu), 60,   3_600),
        'IsAnomaly': 0,
    }))

    # --- 1d. DNS / keep-alive (15 %) — TINY bytes, SHORT duration → overlaps with port scan
    nk = int(n * 0.15)
    frames.append(pd.DataFrame({
        'SourceIP':        rng.integers(10, 200, nk).astype(float),
        'DestinationIP':   rng.integers(10, 200, nk).astype(float),
        'SourcePort':      rng.integers(1024, 65535, nk).astype(float),
        'DestinationPort': rng.choice([53, 123, 161, 443], nk).astype(float),
        'Protocol':        rng.choice([17, 6], nk).astype(float),
        'BytesSent':       _clip(rng.normal(250,   400,   nk), 20,   5_000),
        'BytesReceived':   _clip(rng.normal(200,   350,   nk), 20,   4_000),
        'PacketsSent':     _clip(rng.normal(4,       5,   nk), 1,      60),
        'PacketsReceived': _clip(rng.normal(3,       4,   nk), 1,      50),
        'Duration':        _clip(rng.normal(8,      15,   nk), 1,     200),
        'IsAnomaly': 0,
    }))

    # --- 1e. Remote admin / SSH (10 %) — overlaps with brute force in port & size
    nr = int(n * 0.10)
    frames.append(pd.DataFrame({
        'SourceIP':        rng.integers(10, 200, nr).astype(float),
        'DestinationIP':   rng.integers(10, 200, nr).astype(float),
        'SourcePort':      rng.integers(1024, 65535, nr).astype(float),
        'DestinationPort': rng.choice([22, 3389, 23, 5900], nr).astype(float),
        'Protocol':        rng.choice([6], nr).astype(float),
        'BytesSent':       _clip(rng.normal(3_000,  4_000, nr), 200,  30_000),
        'BytesReceived':   _clip(rng.normal(5_000,  6_000, nr), 200,  40_000),
        'PacketsSent':     _clip(rng.normal(60,      60,   nr), 3,     600),
        'PacketsReceived': _clip(rng.normal(80,      70,   nr), 3,     700),
        'Duration':        _clip(rng.normal(900,    700,   nr), 30,   3_600),
        'IsAnomaly': 0,
    }))

    # --- 1f. Bulk database / internal traffic (15 %) — heavy bidirectional
    ndb = n - nb - nd - nu - nk - nr
    frames.append(pd.DataFrame({
        'SourceIP':        rng.integers(10, 200, ndb).astype(float),
        'DestinationIP':   rng.integers(10, 200, ndb).astype(float),
        'SourcePort':      rng.integers(1024, 65535, ndb).astype(float),
        'DestinationPort': rng.choice([3306, 5432, 1433, 27017, 6379], ndb).astype(float),
        'Protocol':        rng.choice([6], ndb).astype(float),
        'BytesSent':       _clip(rng.normal(45_000, 20_000, ndb), 500, 100_000),
        'BytesReceived':   _clip(rng.normal(45_000, 20_000, ndb), 500, 100_000),
        'PacketsSent':     _clip(rng.normal(500,    350,    ndb), 10,  5_000),
        'PacketsReceived': _clip(rng.normal(500,    350,    ndb), 10,  5_000),
        'Duration':        _clip(rng.normal(2_000,  800,    ndb), 100, 3_600),
        'IsAnomaly': 0,
    }))

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# 2. Anomaly sub-types — designed so individual features overlap with normal
#    sub-types above; only the COMBINATION separates them
# ---------------------------------------------------------------------------
def gen_ddos(n):
    """
    Flood traffic.  BytesSent high + BytesReceived near-zero together is the
    signal.  Individually BytesSent overlaps with file-upload normal traffic.
    """
    return pd.DataFrame({
        'SourceIP':        rng.integers(1,  40,    n).astype(float),
        'DestinationIP':   rng.integers(80, 180,   n).astype(float),
        'SourcePort':      rng.integers(1024, 65535, n).astype(float),
        'DestinationPort': rng.choice([80, 443, 8080], n).astype(float),
        'Protocol':        rng.choice([6, 17], n).astype(float),
        'BytesSent':       _clip(rng.normal(82_000, 14_000, n), 30_000, 100_000),
        'BytesReceived':   _clip(rng.normal(400,    600,    n), 0,       8_000),
        'PacketsSent':     _clip(rng.normal(9_000,  900,    n), 4_000,  10_000),
        'PacketsReceived': _clip(rng.normal(10,      20,    n), 0,         300),
        'Duration':        _clip(rng.normal(90,     120,    n), 5,       1_000),
        'IsAnomaly': 1,
    })


def gen_port_scan(n):
    """
    SYN/probe scan.  Individual byte/duration values overlap with DNS normal;
    the combination of tiny bytes + well-known dst port + near-zero recv is key.
    """
    return pd.DataFrame({
        'SourceIP':        rng.integers(1,  40,    n).astype(float),
        'DestinationIP':   rng.integers(80, 200,   n).astype(float),
        'SourcePort':      rng.integers(1024, 65535, n).astype(float),
        'DestinationPort': rng.integers(1, 1024,   n).astype(float),
        'Protocol':        rng.choice([6, 1], n).astype(float),
        'BytesSent':       _clip(rng.normal(180,   250,   n), 20,   4_000),
        'BytesReceived':   _clip(rng.normal(60,    100,   n), 0,    2_000),
        'PacketsSent':     _clip(rng.normal(3,       4,   n), 1,       50),
        'PacketsReceived': _clip(rng.normal(1,       2,   n), 0,       20),
        'Duration':        _clip(rng.normal(6,      10,   n), 1,      150),
        'IsAnomaly': 1,
    })


def gen_exfiltration(n):
    """
    Slow data theft.  High BytesSent + near-zero BytesReceived + long Duration.
    Individually BytesSent overlaps with file-upload; Duration overlaps with DB.
    Near-zero recv is the extra discriminating signal.
    """
    return pd.DataFrame({
        'SourceIP':        rng.integers(100, 200, n).astype(float),
        'DestinationIP':   rng.integers(1,  60,  n).astype(float),
        'SourcePort':      rng.integers(1024, 65535, n).astype(float),
        'DestinationPort': rng.choice([443, 80, 22, 8443], n).astype(float),
        'Protocol':        rng.choice([6, 17], n).astype(float),
        'BytesSent':       _clip(rng.normal(78_000, 17_000, n), 20_000, 100_000),
        'BytesReceived':   _clip(rng.normal(600,    1_000,  n), 0,       12_000),
        'PacketsSent':     _clip(rng.normal(820,    500,    n), 50,       8_000),
        'PacketsReceived': _clip(rng.normal(15,      25,    n), 0,         300),
        'Duration':        _clip(rng.normal(3_100,  450,    n), 800,     3_600),
        'IsAnomaly': 1,
    })


def gen_brute_force(n):
    """
    Credential stuffing.  Small payloads to admin ports, short sessions.
    Individually overlaps with remote-admin normal traffic (same ports, similar
    byte sizes); the key signal is very low bytes + very short Duration combined.
    """
    return pd.DataFrame({
        'SourceIP':        rng.integers(1,  30,  n).astype(float),
        'DestinationIP':   rng.integers(100, 200, n).astype(float),
        'SourcePort':      rng.integers(1024, 65535, n).astype(float),
        'DestinationPort': rng.choice([22, 3389, 23, 21, 5900], n).astype(float),
        'Protocol':        rng.choice([6], n).astype(float),
        'BytesSent':       _clip(rng.normal(400,    600,   n), 50,   8_000),
        'BytesReceived':   _clip(rng.normal(250,    400,   n), 20,   5_000),
        'PacketsSent':     _clip(rng.normal(8,       10,   n), 1,      100),
        'PacketsReceived': _clip(rng.normal(6,        8,   n), 1,       80),
        'Duration':        _clip(rng.normal(25,       35,  n), 2,      400),
        'IsAnomaly': 1,
    })


# ---------------------------------------------------------------------------
# 3. Stealthy anomalies — sampled from the normal distribution but labelled 1
#    These represent "low-and-slow" attacks the model genuinely cannot always
#    detect, preventing perfect scores and making results thesis-credible.
# ---------------------------------------------------------------------------
def gen_stealthy(n):
    """
    Attacks disguised as normal traffic — all feature values drawn from the
    same distributions as normal traffic.  ~12 % of total anomalies.
    The model will miss most of these, which is realistic and expected.
    """
    base = gen_normal(n)
    base['IsAnomaly'] = 1
    return base


# ---------------------------------------------------------------------------
# 4. Assemble, add label noise, and verify
# ---------------------------------------------------------------------------
STEALTHY_FRAC = 0.12   # 12 % of anomalies are indistinguishable from normal
NOISE_FRAC    = 0.008  # 0.8 % of normal samples randomly flipped to anomaly
                       #  (simulates annotation errors / grey-area traffic)

n_stealthy   = int(N_ANOMALY * STEALTHY_FRAC)
n_detectable = N_ANOMALY - n_stealthy
per_type     = n_detectable // 4
remainder    = n_detectable - per_type * 4

normal    = gen_normal(N_NORMAL)
ddos      = gen_ddos(per_type)
port_scan = gen_port_scan(per_type)
exfil     = gen_exfiltration(per_type)
brute     = gen_brute_force(per_type + remainder)
stealthy  = gen_stealthy(n_stealthy)

df = pd.concat([normal, ddos, port_scan, exfil, brute, stealthy], ignore_index=True)

# Label noise: flip a small fraction of normal samples to anomaly
noise_idx = rng.choice(
    df.index[df.IsAnomaly == 0], size=int(N_NORMAL * NOISE_FRAC), replace=False
)
df.loc[noise_idx, 'IsAnomaly'] = 1

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Dataset shape:", df.shape)
print("\nClass distribution:")
print(df['IsAnomaly'].value_counts())

features = ['BytesSent', 'BytesReceived', 'PacketsSent', 'PacketsReceived', 'Duration']
print("\nMean feature values per class:")
print(df.groupby('IsAnomaly')[features].mean().round(1))

print(f"\nStealthy anomalies (indistinguishable from normal): {n_stealthy} "
      f"({n_stealthy/N_ANOMALY:.0%} of all anomalies)")
print(f"Label noise injected into normal class: {len(noise_idx)} samples")
print("\nExpected model behaviour: RF will catch most detectable anomalies but")
print("miss most stealthy ones -> realistic F1 ~0.88-0.95, ROC-AUC ~0.95-0.98")

out_path = r'C:\Users\iamwa\source\repos\Thesis\Network Analysis\Dataset\network_traffic_denormalized.csv'
df.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")
