# KINO Data Lab

A full-stack statistical analysis and machine learning dashboard for studying high-frequency KINO draw data.

The project imports historical draw results from a public API, stores them in PostgreSQL, builds analytical features, detects board patterns, runs machine learning experiments, performs walk-forward backtesting, and visualizes model results in a React dashboard.

> This project is a data analysis and backtesting experiment. It is not betting advice and does not claim to predict random outcomes with certainty.

---

## Project Status

This project is currently paused, but the existing version already includes:

* A Django backend with PostgreSQL
* Public API data import pipeline
* Incremental sync commands
* Pattern and shape detection
* Machine learning experiments
* Backtesting and ROI simulation
* React / TypeScript dashboard
* Model audit reports
* Near-miss and rescue-strategy analysis

---

## Tech Stack

### Backend

* Python
* Django
* PostgreSQL
* Django management commands
* Scikit-learn
* NumPy
* REST API endpoints

### Frontend

* React
* TypeScript
* Vite
* CSS dashboard UI
* Data visualization components

### Data / Analysis

* Historical draw ingestion
* Sliding-window analysis
* Feature engineering
* Logistic regression experiments
* Walk-forward testing
* ROI simulation
* Model calibration checks
* Pattern and shape detection

---

## Main Features

### 1. Historical Draw Import

The backend imports KINO draw data from the OPAP public API and stores it locally in PostgreSQL.

Supported import workflows include:

* Importing a single day
* Importing a date range
* Syncing latest missing draws
* Running the full processing pipeline

Example commands:

```bash
python manage.py import_kino_range 2026-01-01 2026-06-10
python manage.py sync_kino_latest
```

---

### 2. PostgreSQL Data Storage

Draws are stored in a structured database model, allowing historical analysis, indexing, and repeatable experiments.

Core stored data includes:

* Draw ID
* Draw timestamp
* Drawn numbers
* Analysis state
* Sliding window results
* Pattern events
* Shape events
* AI experiment results

---

### 3. Sliding Window Analysis

The project analyzes recent draw windows to detect short-term and medium-term behavior.

Examples of tracked signals:

* Hot and cold numbers
* Number gaps
* Recent frequency
* Board distribution
* Row and column activity
* Repeated number behavior

---

### 4. Board Pattern Detection

The KINO board is treated as an 8x10 grid.

The system analyzes:

* Row concentration
* Column concentration
* Center / edge activity
* Spread intensity
* Heavy-pattern periods
* Scatter periods
* Quiet/random periods

This creates a higher-level “operation state” for each draw.

---

### 5. Shape Detection

The system detects recurring geometric formations on the number board.

Implemented shape types include:

* Cross
* 2x2 box
* L-shape
* Vertical line
* Horizontal line
* Diagonal down
* Diagonal up

These shapes are used as additional model features and pattern-analysis signals.

---

### 6. Operation Sequence Analysis

The project classifies draw behavior into operation states such as:

* Heavy pattern
* Normal pattern
* Light pattern
* Scatter spread
* Quiet random

It also tracks transitions between states, movement direction, streaks, zones, and center-of-mass movement across the board.

---

### 7. Machine Learning Experiments

Several model versions were tested, starting from simple number-ranking models and moving toward more advanced feature sets.

The later versions include:

* Hot/cold features
* Gap features
* Board-position features
* Row/column pattern features
* Shape features
* Movement features
* Entropy/spread features
* Operation-state features
* Regime-aware selection logic

The goal was not only to predict individual numbers, but also to evaluate whether model rankings were useful under realistic backtesting.

---

### 8. Walk-Forward Backtesting

Instead of testing only on one static split, the project includes walk-forward style testing.

This helps check whether a model performs consistently across different time periods instead of only fitting one historical section.

Tracked metrics include:

* Average hits
* Profit / loss
* ROI
* Hit distribution
* Model lift over baseline
* Performance by regime
* Performance by selection mode

---

### 9. ROI and Payout Simulation

The backend simulates different ticket strategies and payout tables.

Supported analysis includes:

* Cost
* Return
* Profit
* ROI
* Hit distribution
* Dead-zone rounds
* Paying rounds
* Bonus payout logic experiments

This made it possible to compare models not only by hit count, but also by realistic payout behavior.

---

### 10. Model Auditing

The project includes several audit reports to understand model behavior.

Examples:

* Calibration report
* Feature group strength
* High-confidence vs low-confidence performance
* Near-miss analysis
* Rescue-strategy comparison
* Swap-model rescue experiment

These reports helped identify whether the model was genuinely learning useful signals or simply overfitting noise.

---

## Advanced Experiments

### Near-Miss Report

A near-miss analyzer was added to study cases where the model reached 7, 8, or 9 hits.

The report checks whether missing winning numbers were ranked close to the selected combo.

This helped answer questions like:

* Was the missing number inside ranks 13–20?
* Was the model close but the selector skipped the number?
* Did the model fail completely or only fail during final combo construction?

---

### Rescue Selector

A rescue selector was tested to replace weak selected numbers with strong reserve numbers from ranks 13–20.

Tested modes included:

* Rescue 1
* Rescue 2
* Rescue 3
* Smart Rescue 1
* Safe Smart Rescue 1

This experiment showed that simple hand-written rescue rules were not enough and that a learned swap model would be a cleaner next step.

---

### V8 Swap Model Experiment

The final experimental direction was a second-stage swap model.

The idea:

1. Build the base regime-aware combo.
2. Create a reserve pool from ranks 13–20.
3. Generate all one-swap candidates.
4. Calculate the historical profit delta of each swap.
5. Train a model to predict whether a swap improves the result.
6. Apply only the best predicted positive swap.

This turned the selector into a machine-learning problem instead of relying only on manually written rules.

---

## Frontend Dashboard

The React dashboard displays the main analysis results in a portfolio-friendly UI.

Dashboard sections include:

* AI result summaries
* Profit cards
* Best ROI mode
* Latest predicted number sets
* Feature strength
* Calibration results
* Walk-forward performance
* Confidence audits
* Near-miss reports
* Rescue comparison reports
* Swap model results

---

## Example Pipeline

Run the full analysis pipeline:

```bash
python manage.py run_kino_pipeline --rebuild-shapes --rebuild-movements --ai-horizon 10 --ai-decision-step 5 --ai-pick 12 --ai-target-hits 3
```

Run the latest V8 experiment:

```bash
python manage.py train_number_ai_v8 --horizon 10 --decision-step 5 --pick 12 --target-hits 3 --stake 1 --payout-table kino
```

---

## Local Setup

### Backend

```bash
cd backend
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

---

## PostgreSQL Configuration

The project uses a local PostgreSQL database.

Example development configuration:

```python
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": "kino_data_lab",
        "USER": "kino_user",
        "PASSWORD": "kino_password",
        "HOST": "localhost",
        "PORT": "5432",
    }
}
```

---

## What I Learned

This project helped me practice:

* Designing a full-stack data application
* Working with PostgreSQL instead of SQLite
* Building repeatable data import pipelines
* Creating Django management commands
* Structuring machine learning experiments
* Avoiding data leakage in backtesting
* Comparing model results against baselines
* Building React dashboards for complex backend data
* Debugging model results through audit reports
* Thinking critically about noisy datasets and unreliable patterns

---

## Important Notes

This project studies random draw data. The purpose is technical learning, statistical analysis, and full-stack engineering practice.

The project does not claim guaranteed prediction ability or financial profitability.
