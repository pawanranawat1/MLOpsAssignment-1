"""Train, evaluate, track, and package the Heart Disease models."""

import argparse
import json

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from src.config import (
    FIGURES_DIR,
    FEATURE_COLUMNS,
    MODEL_DIR,
    MODEL_PATH,
    METADATA_PATH,
    MLRUNS_DIR,
    MLFLOW_MODEL_NAME,
)
from src.data import load_dataset
from src.preprocess import build_preprocessor


RANDOM_STATE = 42


def evaluate_model(model, x_test, y_test):
    """Calculate the required classification metrics."""
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
    }


def save_plots(model, x_test, y_test, model_name):
    """Save ROC and confusion-matrix plots."""
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_name = model_name.lower().replace(" ", "_")

    RocCurveDisplay.from_estimator(
        model,
        x_test,
        y_test,
    )

    plt.title(f"ROC Curve - {model_name}")
    plt.tight_layout()

    roc_path = (
        FIGURES_DIR /
        f"roc_{safe_name}.png"
    )

    plt.savefig(
        roc_path,
        dpi=150,
    )
    plt.close()

    ConfusionMatrixDisplay.from_estimator(
        model,
        x_test,
        y_test,
    )

    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()

    cm_path = (
        FIGURES_DIR /
        f"confusion_{safe_name}.png"
    )

    plt.savefig(
        cm_path,
        dpi=150,
    )
    plt.close()

    return roc_path, cm_path


def build_models():
    """Create the two required classification pipelines."""

    logistic = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    random_forest = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "model",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    return logistic, random_forest


def main(skip_mlflow: bool = False):
    """Train, evaluate, track, select, and package models."""

    # ---------------------------------------------------------
    # 1. Load dataset
    # ---------------------------------------------------------
    df = load_dataset()

    x = df[FEATURE_COLUMNS]
    y = df["target"]

    # ---------------------------------------------------------
    # 2. Train/test split
    # ---------------------------------------------------------
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # 3. Configure MLflow
    # ---------------------------------------------------------
    if not skip_mlflow:
        mlflow.set_tracking_uri(
            MLRUNS_DIR
        )

        mlflow.set_experiment(
            "heart-disease-classification"
        )

    # ---------------------------------------------------------
    # 4. Build the two base models
    # ---------------------------------------------------------
    logistic, random_forest = build_models()

    results = []

    trained_models = {}

    # ---------------------------------------------------------
    # 5. Train and evaluate base models
    # ---------------------------------------------------------
    for name, model in zip(
        [
            "Logistic Regression",
            "Random Forest",
        ],
        [
            logistic,
            random_forest,
        ],
    ):

        run_context = (
            mlflow.start_run(
                run_name=name
            )
            if not skip_mlflow
            else None
        )

        if run_context:
            run_context.__enter__()

        if not skip_mlflow:
            mlflow.set_tag(
                "purpose",
                f"Evaluate {name} classification model",
            )

        try:
            model.fit(
                x_train,
                y_train,
            )

            trained_models[name] = model

            metrics = evaluate_model(
                model,
                x_test,
                y_test,
            )

            roc_path, cm_path = save_plots(
                model,
                x_test,
                y_test,
                name,
            )

            if not skip_mlflow:
                mlflow.log_param(
                    "model",
                    name,
                )

                mlflow.log_param(
                    "test_size",
                    0.20,
                )

                mlflow.log_param(
                    "random_state",
                    RANDOM_STATE,
                )

                mlflow.log_metrics(
                    metrics
                )

                mlflow.log_artifact(
                    str(roc_path)
                )

                mlflow.log_artifact(
                    str(cm_path)
                )

            row = {
                "model": name,
                **metrics,
            }

            results.append(row)

            print(row)

        finally:
            if run_context:
                run_context.__exit__(
                    None,
                    None,
                    None,
                )

    # ---------------------------------------------------------
    # 6. Tune Random Forest using 5-fold cross-validation
    # ---------------------------------------------------------
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    rf_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "model",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    param_grid = {
        "model__n_estimators": [
            100,
            200,
        ],
        "model__max_depth": [
            None,
            5,
            10,
        ],
        "model__min_samples_split": [
            2,
            5,
        ],
    }

    tuned = GridSearchCV(
        rf_pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
    )

    run_context = (
        mlflow.start_run(
            run_name="Random Forest Tuned"
        )
        if not skip_mlflow
        else None
    )

    if run_context:
        run_context.__enter__()

    if not skip_mlflow:
        mlflow.set_tag(
            "purpose",
            "Evaluate hyperparameter-tuned Random Forest model",
        )

    try:
        tuned.fit(
            x_train,
            y_train,
        )

        tuned_metrics = evaluate_model(
            tuned,
            x_test,
            y_test,
        )

        roc_path, cm_path = save_plots(
            tuned,
            x_test,
            y_test,
            "Random Forest Tuned",
        )

        if not skip_mlflow:
            mlflow.log_params(
                tuned.best_params_
            )

            mlflow.log_metric(
                "cv_best_roc_auc",
                tuned.best_score_,
            )

            mlflow.log_metrics(
                tuned_metrics
            )

            mlflow.log_artifact(
                str(roc_path)
            )

            mlflow.log_artifact(
                str(cm_path)
            )

        results.append(
            {
                "model": "Random Forest Tuned",
                **tuned_metrics,
            }
        )

        trained_models[
            "Random Forest Tuned"
        ] = tuned.best_estimator_

    finally:
        if run_context:
            run_context.__exit__(
                None,
                None,
                None,
            )

    # ---------------------------------------------------------
    # 7. Model comparison
    # ---------------------------------------------------------
    results_df = (
        pd.DataFrame(results)
        .sort_values(
            "roc_auc",
            ascending=False,
        )
    )

    results_df.to_csv(
        FIGURES_DIR /
        "model_comparison.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 8. Select the best model dynamically
    #
    # The current assignment evaluation uses ROC-AUC.
    # Therefore the model with the highest held-out
    # ROC-AUC is selected.
    # ---------------------------------------------------------
    best_result = results_df.iloc[0]

    best_model_name = best_result["model"]

    final_model = trained_models[
        best_model_name
    ]

    # ---------------------------------------------------------
    # 9. Retrain selected model on all available data
    # ---------------------------------------------------------
    final_model.fit(
        x,
        y,
    )

    # ---------------------------------------------------------
    # 10. Save reusable local model copy
    # ---------------------------------------------------------
    joblib.dump(
        final_model,
        MODEL_PATH,
    )

    # ---------------------------------------------------------
    # 11. Save metadata
    # ---------------------------------------------------------
    metadata = {
        "model_name": best_model_name,
        "target": "target",
        "features": FEATURE_COLUMNS,
        "binary_target_definition": (
            "0 = no disease, 1 = disease "
            "(original UCI target > 0)"
        ),
        "selection_metric": "test_roc_auc",
        "test_metrics": {
            key: float(best_result[key])
            for key in [
                "accuracy",
                "precision",
                "recall",
                "roc_auc",
            ]
        },
        "training_rows": int(len(x)),
        "random_state": RANDOM_STATE,
    }

    if best_model_name == "Random Forest Tuned":
        metadata[
            "best_parameters"
        ] = tuned.best_params_

        metadata[
            "cv_best_roc_auc"
        ] = float(
            tuned.best_score_
        )

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )

    # ---------------------------------------------------------
    # 12. Register final model with MLflow
    # ---------------------------------------------------------
    if not skip_mlflow:

        with mlflow.start_run(
            run_name="Final Model Packaging"
        ):

            mlflow.set_tag(
                "purpose",
                "Package and register the dynamically selected best model",
            )

            mlflow.log_param(
                "selected_model",
                best_model_name,
            )

            mlflow.log_param(
                "selection_metric",
                "test_roc_auc",
            )

            mlflow.log_metric(
                "selected_test_accuracy",
                float(best_result["accuracy"]),
            )

            mlflow.log_metric(
                "selected_test_precision",
                float(best_result["precision"]),
            )

            mlflow.log_metric(
                "selected_test_recall",
                float(best_result["recall"]),
            )

            mlflow.log_metric(
                "selected_test_roc_auc",
                float(best_result["roc_auc"]),
            )

            mlflow.sklearn.log_model(
                sk_model=final_model,
                artifact_path="model",
                registered_model_name=MLFLOW_MODEL_NAME,
            )

    # ---------------------------------------------------------
    # 13. Final output
    # ---------------------------------------------------------
    print("\nModel comparison:")
    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        f"\nSelected model: "
        f"{best_model_name}"
    )

    print(
        f"Saved model: "
        f"{MODEL_PATH}"
    )

    print(
        f"Saved metadata: "
        f"{METADATA_PATH}"
    )

    if not skip_mlflow:
        print(
            f"Registered MLflow model: "
            f"{MLFLOW_MODEL_NAME}"
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--skip-mlflow",
        action="store_true",
        help=(
            "Train without creating "
            "MLflow runs. Useful for "
            "quick local testing."
        ),
    )

    args = parser.parse_args()

    main(
        skip_mlflow=args.skip_mlflow
    )
