# PolySense QuAM 

## Description

PolySense QuAM is a query answering machine that uses machine learning models to predict stock price movements and Polymarket outcomes. It supports two prediction phases: Phase A predicts Polymarket direction outcomes, while Phase B predicts whether stock prices will close higher than the previous close. The system uses scikit-learn classifiers (Logistic Regression, Random Forest, and Gradient Boosting) trained on multimodal feature sets with over 80% accuracy.

The complete code for creating and training models is included in the notebooks under `notebooks/`, including data wrangling, feature engineering, and model iteration experiments.

## Installation

PolySense QuAM requires Python 3.10 or above, which can be downloaded from the [official Python webpage](https://www.python.org/downloads/).

Various Python libraries are required to run the program (see `requirements.txt` for the complete list). Installation can be done in a Python virtual environment or in your local Python installation.

### Installation with Docker (Recommended)

No additional setup needed beyond Docker. Run:

```bash
./scripts/run_all.sh
```

This automatically detects `docker compose` or `docker-compose` and deploys the entire application in one container.

### Installation in local Python

Run the following command:

```bash
pip install -r requirements.txt
```

### Installation in a virtual environment

Navigate to your desired directory and run:

```bash
python -m venv venv
```

On Windows, activate with:
```bash
venv\Scripts\activate
```

On Linux/Mac, activate with:
```bash
source venv/bin/activate
```

Then install dependencies:
```bash
pip install -r requirements.txt
```

To exit the virtual environment, run:
```bash
deactivate
```

## How to run

### With Docker (Recommended)

From the repo root:

```bash
./scripts/run_all.sh
```

This starts both the frontend and backend in a single container:
- Frontend: `http://127.0.0.1:5173`
- Backend API: `http://127.0.0.1:8000`

To stop the application:
```bash
./scripts/stop_all.sh
```

### Local setup (without Docker)

In the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
./scripts/run_backend.sh
```

In a second terminal:
```bash
./scripts/run_frontend.sh
```

Then open `http://127.0.0.1:5173`

### Using the interface

1. **Dataset input**: Upload a CSV, register a dataset path, or select from pre-loaded datasets.
2. **Phase selection**: Choose Phase A (Polymarket direction) or Phase B (stock close direction).
3. **Parameter configuration**: Set target column, exclusions, and feature engineering options.
4. **Model training**: Adjust hyperparameters and train the Gradient Boosting model with cross-validation.
5. **Inference**: Run single or batch predictions using the trained model.

## Dataset

PolySense QuAM works with multimodal datasets containing stock price features, technical indicators, and sentiment data.

- **Phase A datasets**: Require a `pm_direction_up` or `pm_direction_up_final` column indicating Polymarket up/down direction.
- **Phase B datasets**: Require an `end_price` column to automatically derive whether the stock closed higher than the previous close.

Pre-registered datasets are located in `data/processed/`. Custom datasets can be uploaded or registered via the web interface. The system automatically handles malformed CSV lines and performs robust feature engineering.


## About

PolySense QuAM is a machine learning project that combines multiple data sources (stock prices, sentiment analysis, and market data) to predict financial outcomes. Using scikit-learn classifiers trained with repeated stratified k-fold cross-validation, the system achieves high accuracy in both Polymarket direction and stock price movement prediction. The minimalist web interface decouples model training from inference, allowing rapid iteration and easy integration into larger prediction pipelines.

## Topics

machine-learning, prediction, polymarket, stock-prediction, FASTapi, scikit-learn, time-series
