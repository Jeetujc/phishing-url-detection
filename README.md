# Phishing URL Detection

A machine-learning web application that classifies URLs as **safe (legitimate)** or **unsafe (phishing)** using an ensemble classifier trained on 30 hand-crafted features.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [How the Model Works](#how-the-model-works)
3. [Training a New Model](#training-a-new-model)
   - [Using the Jupyter Notebook](#using-the-jupyter-notebook)
   - [Using the Training Script](#using-the-training-script)
4. [Preparing Your Own Dataset](#preparing-your-own-dataset)
5. [Generating Synthetic Data](#generating-synthetic-data)
6. [Model Performance](#model-performance)
7. [Feature Reference](#feature-reference)
8. [Project Structure](#project-structure)

---

## Quick Start

```bash
# 1. Clone & install dependencies
pip install -r requirements.txt

# 2. Run the Flask app
python app.py
# → open http://127.0.0.1:5000
```

---

## How the Model Works

Feature extraction (`feature.py`) computes **30 binary/ternary features** from a URL:

| Value | Meaning |
|-------|---------|
| `1`   | Legitimate indicator |
| `0`   | Suspicious / neutral |
| `-1`  | Phishing indicator |

The classifier (`pickle/model.pkl`) maps these 30 features to:

| Prediction | Meaning |
|-----------|---------|
| `1`       | Safe / Legitimate |
| `-1`      | Unsafe / Phishing |

---

## Training a New Model

### Using the Jupyter Notebook

The notebook (`train_model.ipynb`) provides an **interactive** training experience with visualizations:

```bash
jupyter notebook train_model.ipynb
```

The notebook covers:
- Dataset exploration and class balance analysis
- Feature distribution plots by class
- Feature correlation heatmap
- Training **4 models**: Gradient Boosting, Random Forest, XGBoost, LightGBM
- 5-fold stratified cross-validation
- Hyperparameter tuning with `GridSearchCV`
- Confusion matrices and ROC curves
- Feature importance comparison
- Saving the best model to `pickle/model.pkl`

### Using the Training Script

For automated / scheduled retraining:

```bash
# Train on the default dataset (phishing.csv), with hyperparameter tuning
python train.py

# Train on a custom dataset
python train.py --data my_urls.csv

# Skip hyperparameter tuning (faster)
python train.py --no-tune

# Use 10-fold cross-validation
python train.py --cv-folds 10
```

The script:
1. Loads the CSV dataset
2. Trains GBC, Random Forest, XGBoost, and LightGBM
3. Cross-validates each model (5-fold by default)
4. Selects the best model by test accuracy
5. Optionally tunes hyperparameters with `GridSearchCV`
6. Saves the model to `pickle/model.pkl`
7. Saves a JSON performance report to `training_report.json`

---

## Preparing Your Own Dataset

Your CSV must have the following columns (in any order):

```
Index, UsingIP, LongURL, ShortURL, Symbol@, Redirecting//,
PrefixSuffix-, SubDomains, HTTPS, DomainRegLen, Favicon,
NonStdPort, HTTPSDomainURL, RequestURL, AnchorURL,
LinksInScriptTags, ServerFormHandler, InfoEmail, AbnormalURL,
WebsiteForwarding, StatusBarCust, DisableRightClick,
UsingPopupWindow, IframeRedirection, AgeofDomain, DNSRecording,
WebsiteTraffic, PageRank, GoogleIndex, LinksPointingToPage,
StatsReport, class
```

- All feature columns take values in `{-1, 0, 1}`
- The `class` column must be `1` (legitimate) or `-1` (phishing)

---

## Generating Synthetic Data

If you don't have a real dataset, generate synthetic data for testing:

```bash
# Generate 2000 balanced samples (default)
python generate_sample_data.py

# Generate 5000 samples with 60% phishing
python generate_sample_data.py --samples 5000 --phishing-ratio 0.6 --out training_data.csv

# Options
python generate_sample_data.py --help
```

---

## Model Performance

Results on the included `phishing.csv` dataset (11,054 samples, 80/20 train/test split):

| Model | CV Accuracy | Test Accuracy | F1 Score | ROC-AUC |
|-------|------------|--------------|---------|---------|
| Gradient Boosting | 0.9594 ± 0.006 | 0.9661 | 0.9661 | 0.9950 |
| **Random Forest** | **0.9707 ± 0.003** | **0.9738** | **0.9737** | 0.9945 |
| XGBoost | 0.9619 ± 0.004 | 0.9670 | 0.9670 | 0.9951 |
| LightGBM | 0.9648 ± 0.004 | 0.9697 | 0.9697 | **0.9956** |

> **Random Forest** is selected as the default best model (highest test accuracy).  
> Run `train.py` to automatically pick the best model for your own data.

---

## Feature Reference

| # | Feature | Description |
|---|---------|-------------|
| 1 | UsingIP | IP address used instead of domain name |
| 2 | LongURL | URL length > 75 chars → phishing |
| 3 | ShortURL | Known URL shortener service |
| 4 | Symbol@ | `@` symbol present in URL |
| 5 | Redirecting// | `//` appears after position 7 |
| 6 | PrefixSuffix- | `-` present in domain name |
| 7 | SubDomains | Number of subdomains |
| 8 | HTTPS | HTTPS protocol in URL |
| 9 | DomainRegLen | Domain registration length < 1 year |
| 10 | Favicon | Favicon loaded from external domain |
| 11 | NonStdPort | Non-standard port in URL |
| 12 | HTTPSDomainURL | "https" token appears in domain |
| 13 | RequestURL | % of page resources from external domains |
| 14 | AnchorURL | % of anchor links to external domains |
| 15 | LinksInScriptTags | % of script/link tags to external domains |
| 16 | ServerFormHandler | Form action points to external domain |
| 17 | InfoEmail | Page uses `mailto:` links |
| 18 | AbnormalURL | URL domain doesn't match WHOIS record |
| 19 | WebsiteForwarding | Number of HTTP redirects |
| 20 | StatusBarCust | Mouseover JS changes status bar |
| 21 | DisableRightClick | Right-click disabled via JS |
| 22 | UsingPopupWindow | JS popup window with text field |
| 23 | IframeRedirection | iframe present in page |
| 24 | AgeofDomain | Domain age < 6 months (from WHOIS) |
| 25 | DNSRecording | DNS record age < 6 months |
| 26 | WebsiteTraffic | Tranco traffic rank (replaces Alexa) |
| 27 | PageRank | Open PageRank score (replaces checkpagerank.net) |
| 28 | GoogleIndex | Domain resolves via DNS (proxy for indexing) |
| 29 | LinksPointingToPage | Number of external links pointing to page |
| 30 | StatsReport | Domain/IP matches known phishing lists |

---

## Project Structure

```
phishing-url-detection/
├── app.py                    # Flask web application
├── feature.py                # Feature extraction (30 features per URL)
├── model_utils.py            # Shared model wrapper (LabelDecodingWrapper)
├── train.py                  # Automated training pipeline
├── train_model.ipynb         # Interactive Jupyter notebook
├── generate_sample_data.py   # Synthetic dataset generator
├── phishing.csv              # Original labeled dataset (11,054 rows)
├── requirements.txt          # Python dependencies
├── pickle/
│   └── model.pkl             # Trained model (compatible with app.py)
├── static/                   # CSS / JS assets
└── templates/
    └── index.html            # Web UI template
```
