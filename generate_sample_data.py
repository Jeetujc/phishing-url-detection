"""
generate_sample_data.py
-----------------------
Creates a synthetic dataset of phishing and legitimate URL feature vectors
in the same format as phishing.csv (features already extracted, no live HTTP
requests required).

Labels follow the original convention used by the pre-trained model:
    1  -> legitimate / safe
   -1  -> phishing / unsafe

Usage
-----
    python generate_sample_data.py                   # writes dataset.csv
    python generate_sample_data.py --out mydata.csv  # custom output path
    python generate_sample_data.py --samples 5000    # generate 5000 rows
"""

import argparse
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Feature column names – must match the order expected by the model
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "UsingIP", "LongURL", "ShortURL", "Symbol@", "Redirecting//",
    "PrefixSuffix-", "SubDomains", "HTTPS", "DomainRegLen", "Favicon",
    "NonStdPort", "HTTPSDomainURL", "RequestURL", "AnchorURL",
    "LinksInScriptTags", "ServerFormHandler", "InfoEmail", "AbnormalURL",
    "WebsiteForwarding", "StatusBarCust", "DisableRightClick",
    "UsingPopupWindow", "IframeRedirection", "AgeofDomain", "DNSRecording",
    "WebsiteTraffic", "PageRank", "GoogleIndex", "LinksPointingToPage",
    "StatsReport",
]

# ---------------------------------------------------------------------------
# Feature value distributions derived from published phishing datasets
# (UCI / PhishTank research papers).
# Each feature takes one of {-1, 0, 1}; probabilities are [P(-1), P(0), P(1)].
# ---------------------------------------------------------------------------

# Probabilities for phishing URLs
PHISHING_PROBS = {
    "UsingIP":            [0.30, 0.05, 0.65],
    "LongURL":            [0.60, 0.25, 0.15],
    "ShortURL":           [0.40, 0.05, 0.55],
    "Symbol@":            [0.30, 0.05, 0.65],
    "Redirecting//":      [0.35, 0.05, 0.60],
    "PrefixSuffix-":      [0.70, 0.05, 0.25],
    "SubDomains":         [0.55, 0.25, 0.20],
    "HTTPS":              [0.60, 0.05, 0.35],
    "DomainRegLen":       [0.75, 0.05, 0.20],
    "Favicon":            [0.55, 0.05, 0.40],
    "NonStdPort":         [0.25, 0.05, 0.70],
    "HTTPSDomainURL":     [0.20, 0.05, 0.75],
    "RequestURL":         [0.55, 0.25, 0.20],
    "AnchorURL":          [0.65, 0.20, 0.15],
    "LinksInScriptTags":  [0.55, 0.25, 0.20],
    "ServerFormHandler":  [0.50, 0.20, 0.30],
    "InfoEmail":          [0.40, 0.05, 0.55],
    "AbnormalURL":        [0.70, 0.05, 0.25],
    "WebsiteForwarding":  [0.45, 0.25, 0.30],
    "StatusBarCust":      [0.45, 0.05, 0.50],
    "DisableRightClick":  [0.30, 0.05, 0.65],
    "UsingPopupWindow":   [0.25, 0.05, 0.70],
    "IframeRedirection":  [0.35, 0.05, 0.60],
    "AgeofDomain":        [0.65, 0.05, 0.30],
    "DNSRecording":       [0.65, 0.05, 0.30],
    "WebsiteTraffic":     [0.60, 0.10, 0.30],
    "PageRank":           [0.65, 0.05, 0.30],
    "GoogleIndex":        [0.50, 0.05, 0.45],
    "LinksPointingToPage":[0.45, 0.25, 0.30],
    "StatsReport":        [0.35, 0.05, 0.60],
}

# Probabilities for legitimate URLs (roughly inverted)
LEGIT_PROBS = {
    "UsingIP":            [0.02, 0.03, 0.95],
    "LongURL":            [0.10, 0.20, 0.70],
    "ShortURL":           [0.05, 0.05, 0.90],
    "Symbol@":            [0.02, 0.03, 0.95],
    "Redirecting//":      [0.05, 0.05, 0.90],
    "PrefixSuffix-":      [0.15, 0.05, 0.80],
    "SubDomains":         [0.10, 0.30, 0.60],
    "HTTPS":              [0.10, 0.05, 0.85],
    "DomainRegLen":       [0.10, 0.05, 0.85],
    "Favicon":            [0.10, 0.05, 0.85],
    "NonStdPort":         [0.05, 0.05, 0.90],
    "HTTPSDomainURL":     [0.05, 0.05, 0.90],
    "RequestURL":         [0.05, 0.20, 0.75],
    "AnchorURL":          [0.10, 0.30, 0.60],
    "LinksInScriptTags":  [0.05, 0.30, 0.65],
    "ServerFormHandler":  [0.05, 0.20, 0.75],
    "InfoEmail":          [0.05, 0.05, 0.90],
    "AbnormalURL":        [0.10, 0.05, 0.85],
    "WebsiteForwarding":  [0.05, 0.20, 0.75],
    "StatusBarCust":      [0.10, 0.05, 0.85],
    "DisableRightClick":  [0.05, 0.05, 0.90],
    "UsingPopupWindow":   [0.05, 0.05, 0.90],
    "IframeRedirection":  [0.10, 0.05, 0.85],
    "AgeofDomain":        [0.10, 0.05, 0.85],
    "DNSRecording":       [0.10, 0.05, 0.85],
    "WebsiteTraffic":     [0.10, 0.20, 0.70],
    "PageRank":           [0.10, 0.10, 0.80],
    "GoogleIndex":        [0.05, 0.05, 0.90],
    "LinksPointingToPage":[0.05, 0.15, 0.80],
    "StatsReport":        [0.05, 0.05, 0.90],
}

VALUES = [-1, 0, 1]


def generate_samples(n_samples: int, label: int, probs: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Return a DataFrame of *n_samples* rows with features drawn from *probs*."""
    data = {}
    for col in FEATURE_COLUMNS:
        p = probs[col]
        data[col] = rng.choice(VALUES, size=n_samples, p=p)
    df = pd.DataFrame(data)
    df["class"] = label
    return df


def generate_dataset(n_samples: int = 2000, phishing_ratio: float = 0.5,
                     seed: int = 42) -> pd.DataFrame:
    """
    Generate a balanced (or custom-ratio) synthetic dataset.

    Parameters
    ----------
    n_samples     : total number of rows
    phishing_ratio: fraction of phishing samples (default 0.5 = balanced)
    seed          : random seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    n_phishing = int(n_samples * phishing_ratio)
    n_legit = n_samples - n_phishing

    df_phish = generate_samples(n_phishing, label=-1, probs=PHISHING_PROBS, rng=rng)
    df_legit = generate_samples(n_legit,   label=1,  probs=LEGIT_PROBS,    rng=rng)

    df = pd.concat([df_phish, df_legit], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle
    df.index.name = "Index"
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate a synthetic phishing URL dataset.")
    parser.add_argument("--out",     default="dataset.csv", help="Output CSV file path (default: dataset.csv)")
    parser.add_argument("--samples", type=int, default=2000, help="Total number of samples (default: 2000)")
    parser.add_argument("--phishing-ratio", type=float, default=0.5,
                        help="Fraction of phishing samples (default: 0.5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    print(f"Generating {args.samples} samples "
          f"({int(args.samples * args.phishing_ratio)} phishing / "
          f"{args.samples - int(args.samples * args.phishing_ratio)} legitimate) …")

    df = generate_dataset(
        n_samples=args.samples,
        phishing_ratio=args.phishing_ratio,
        seed=args.seed,
    )

    df.to_csv(args.out)
    print(f"Dataset saved to '{args.out}'  ({len(df)} rows, {df.columns.tolist()})")
    print("\nClass distribution:")
    print(df["class"].value_counts().rename({1: "legitimate (1)", -1: "phishing (-1)"}))


if __name__ == "__main__":
    main()
