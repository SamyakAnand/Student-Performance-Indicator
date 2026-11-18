# Student Performance Indicator

An end-to-end regression project that explores how demographic and educational context relates to student mathematics scores. It includes exploratory analysis, preprocessing, model training, a Flask prediction interface, and a Docker deployment configuration.

## Project output

<div align="center">
  <img src="Images/index.png" alt="Application landing page" width="280">
  <img src="Images/home.png" alt="Student feature input form" width="280">
  <img src="Images/result.png" alt="Prediction result page" width="280">
</div>

## Problem and dataset

The project uses the [Students Performance in Exams dataset on Kaggle](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams).

- **Dataset size:** 1,000 rows and 8 columns.
- **Target:** mathematics score (**math_score**).
- **Predictors:** gender, race/ethnicity, parental education, lunch type, test preparation, reading score, and writing score.

The dataset is educational and observational. The model estimates a score from the supplied features; it does not establish causal effects or measure a student's overall ability.

## Workflow

1. **Data ingestion:** read the source CSV and create reproducible train/test partitions.
2. **Transformation:** impute missing values, encode categorical features, and scale numeric features.
3. **Model selection:** compare regression estimators and tune hyperparameters using 3-fold cross-validation on the training partition.
4. **Final evaluation:** use the held-out test partition only after model selection, for a final R² report.
5. **Prediction:** load the saved preprocessing and model artifacts for the Flask interface.

The training code keeps model and hyperparameter selection separate from final holdout evaluation. The committed model artifact is not regenerated automatically by a code change; rerun the training pipeline to create an artifact using the revised evaluation protocol.

**Cross-validation preprocessing caveat:** The current transformation component fits imputation, encoding, and scaling on the complete training partition before GridSearchCV runs. The holdout test partition remains untouched, but the cross-validation folds therefore share preprocessing statistics and their scores may be optimistic. A follow-up should place the preprocessing transformer inside the estimator Pipeline searched by GridSearchCV, so each fold fits preprocessing only on its own training split.

## Local setup

Use Python 3.11 for the automated test environment.

    python -m venv .venv
    pip install -r requirements.txt
    python app.py

The Flask application listens on port 5000 by default. The repository also includes a Dockerfile and a Render configuration; review the deployment settings and required environment before deploying.

## Run tests

The focused test dependencies are separate from the full model-training stack:

    pip install -r requirements-test.txt
    pytest -q

GitHub Actions runs the test suite on pull requests and pushes to main. The tests cover the model-selection helper using small synthetic regression datasets; they do not retrain every estimator in the full training grid.

## Repository structure

    .
    ├── .github/workflows/       # Pull-request and push CI
    ├── artifacts/               # Dataset splits and serialized model artifacts
    ├── notebook/                 # EDA and model-training notebooks
    ├── src/
    │   ├── components/          # Ingestion, transformation, model training
    │   ├── pipeline/             # Training and prediction pipelines
    │   └── utils.py               # Serialization and model evaluation
    ├── templates/                # Flask HTML templates
    ├── static/                   # Application styles
    ├── app.py                    # Flask entry point
    ├── requirements.txt
    └── requirements-test.txt

## Limitations

- This is an educational regression project, not an assessment or decision system for schools.
- The dataset is a small, specific sample and may not generalize to other populations or educational settings.
- Cross-validation scores and the final holdout score are dataset-dependent; no performance claim is made here without a fresh training run.
- Existing serialized artifacts should be retrained and evaluated before being described as products of the revised selection protocol.

## Dataset attribution

Please consult the [original Kaggle dataset page](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams) for its source, contributors, and usage terms.
