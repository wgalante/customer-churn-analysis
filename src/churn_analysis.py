"""Customer churn analysis module."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split


def load_data(dataset_path: str) -> pd.DataFrame:
    """Load churn data from a CSV file."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset não encontrado em: {dataset_path}")

    df = pd.read_csv(dataset_path)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Convert TotalCharges to numeric and clean rows with missing values."""
    df = df.copy()

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    missing_total = df['TotalCharges'].isna().sum()
    if missing_total > 0:
        df = df.dropna(subset=['TotalCharges'])

    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical variables using one-hot encoding."""
    df = df.copy()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    categorical_columns = [col for col in categorical_columns if col != 'customerID']

    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    return df


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare feature matrix and target vector for modeling."""
    if 'Churn' not in df.columns:
        raise KeyError('A coluna Churn deve estar presente no DataFrame.')

    X = df.drop(columns=['customerID', 'Churn'])
    y = df['Churn']
    return X, y


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> LogisticRegression:
    """Train a logistic regression model on the churn dataset."""
    model = LogisticRegression(max_iter=200, solver='liblinear', random_state=random_state)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: LogisticRegression, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Compute evaluation metrics for the trained model."""
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, output_dict=True)
    matrix = confusion_matrix(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)

    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': matrix,
    }


def save_plots(df: pd.DataFrame, output_path: str) -> None:
    """Generate and save the four main plots used in the analysis."""
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    churn_rate = df['Churn'].value_counts(normalize=True) * 100
    churn_rate.plot(kind='bar', color=['#4c72b0', '#dd8452'])
    plt.title('Taxa geral de churn')
    plt.xticks(ticks=[0, 1], labels=['Não churn', 'Churn'], rotation=0)
    plt.ylabel('Porcentagem (%)')
    plt.tight_layout()
    plt.savefig(output_dir / 'churn_rate.png')
    plt.close()

    plt.figure(figsize=(8, 5))
    contract_churn = df.groupby('Contract')['Churn'].mean().sort_values(ascending=False)
    contract_churn.plot(kind='bar', color='#55a868')
    plt.title('Churn por tipo de contrato')
    plt.ylabel('Taxa de churn')
    plt.tight_layout()
    plt.savefig(output_dir / 'churn_by_contract.png')
    plt.close()

    plt.figure(figsize=(8, 5))
    churn_tenure = df[df['Churn'] == 1]['tenure']
    plt.hist(churn_tenure, bins=20, color='#c44e52')
    plt.title('Distribuição de tenure para clientes com churn')
    plt.xlabel('Tempo de cliente (meses)')
    plt.ylabel('Contagem')
    plt.tight_layout()
    plt.savefig(output_dir / 'churn_by_tenure.png')
    plt.close()

    plt.figure(figsize=(8, 5))
    df.boxplot(column='MonthlyCharges', by='Churn', grid=False)
    plt.title('MonthlyCharges por churn')
    plt.suptitle('')
    plt.xlabel('Churn')
    plt.ylabel('MonthlyCharges')
    plt.xticks(ticks=[1, 2], labels=['Não churn', 'Churn'])
    plt.tight_layout()
    plt.savefig(output_dir / 'monthly_charges_boxplot.png')
    plt.close()


def main() -> None:
    """Run the churn analysis pipeline end to end."""
    dataset_path = Path(__file__).resolve().parents[1] / 'data' / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
    output_dir = Path(__file__).resolve().parents[1] / 'images'

    df = load_data(str(dataset_path))
    df_clean = preprocess_data(df)
    save_plots(df_clean, str(output_dir))

    df_encoded = encode_features(df_clean)
    X, y = build_feature_matrix(df_encoded)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = train_logistic_regression(X_train, y_train)
    results = evaluate_model(model, X_test, y_test)

    print(f"Accuracy: {results['accuracy']:.4f}")
    print('Classification report:')
    print(pd.DataFrame(results['classification_report']).transpose())
    print('Confusion matrix:')
    print(results['confusion_matrix'])


if __name__ == '__main__':
    main()
