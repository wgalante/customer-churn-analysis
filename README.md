# Customer Churn Prediction — Telco Revenue Protection

> **Predictive modeling project identifying $16,669/month in at-risk recurring revenue across 7,043 telecom customers.**

![Churn Prediction Infographic](images/telco_churn.png)

---

## 📋 Table of Contents

- [Business Problem](#-business-problem)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Key Findings](#-key-findings)
- [Model Results](#-model-results)
- [Business Impact](#-business-impact)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [How to Run](#-how-to-run)

---

## 🎯 Business Problem

In the telecom industry, **acquiring a new customer costs 5–7x more than retaining an existing one**. With a baseline churn rate of 26.5% across 7,043 customers, undetected cancellations represent compounded revenue loss that compounds monthly.

**The core question this project answers:**

> *"Given a customer's current profile, what is the probability they will cancel their service in the coming months — and how much revenue is at risk?"*

Answering this enables the business to shift from **expensive reactive win-backs** to **low-cost proactive interventions** — offers, renegotiations, or service improvements — before the customer decides to leave.

---

## 📦 Dataset

**Source:** [IBM Watson Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — publicly available on Kaggle.

| Attribute | Details |
|---|---|
| Records | 7,043 customers |
| Features | 21 columns |
| Target variable | `Churn` — Yes / No |
| Coverage | Demographics, services, account info, billing |

**Feature categories:**
- **Demographics:** gender, senior citizen status, dependents
- **Services:** internet, phone, streaming, tech support
- **Account:** contract type, payment method, monthly/total charges
- **Target:** whether the customer churned (Yes/No)

---

## 🔬 Methodology

### Step 1 — Exploratory Data Analysis (EDA)

- Profiled 7,043 records for inconsistencies and missing values
- Identified `TotalCharges` stored as string instead of numeric — converted with `pd.to_numeric(errors='coerce')`
- Removed 11 records with null values after conversion (0.15% data loss — acceptable)
- Analyzed distributions, correlations, and churn patterns across all feature groups

### Step 2 — Feature Engineering

**One-Hot Encoding** for categorical variables:
```python
df = pd.get_dummies(df, drop_first=True)
```
`drop_first=True` removes one dummy column per feature to avoid multicollinearity.

**StandardScaler** for numerical features:
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
```
Ensures `MonthlyCharges` ($18–$119) and `Tenure` (0–72 months) operate on the same scale, preventing the model from assigning disproportionate weight to larger values.

### Step 3 — Model Selection

**Logistic Regression** was chosen over more complex alternatives (Random Forest, XGBoost) for three deliberate reasons:

| Reason | Rationale |
|---|---|
| **Interpretability** | Each coefficient maps directly to a business lever a non-technical manager can act on |
| **Speed** | Trains in seconds, enabling rapid iteration and hypothesis validation |
| **Strong baseline** | Solves the problem well without unnecessary complexity |

### Step 4 — Evaluation Strategy

**Recall was prioritized over Accuracy** because the dataset is imbalanced (73.5% non-churn) and the cost asymmetry favors catching churners:

- **False Negative** (missed churner) = lost recurring revenue — high cost
- **False Positive** (unnecessary intervention) = cost of a retention offer — low and controllable

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
```

---

## 📊 Key Findings

### 1. Contract Type is the Strongest Predictor

| Contract Type | Churn Rate |
|---|---|
| Month-to-Month | **42%** |
| One Year | 11% |
| Two Year | **3%** |

Month-to-month customers churn at **14x the rate** of two-year contract holders. A contract migration strategy is more effective than generic discounts.

### 2. The 10-Month Danger Zone

**67% of all cancellations happen within the first 10 months** of tenure. The onboarding period is the highest-risk window — early value delivery and engagement are critical.

### 3. Price-Value Mismatch

Customers who churned paid a **median of $79/month**, compared to $64 for retained customers — **23% more**. This signals a perceived value problem, not a pricing problem.

---

## 📈 Model Results

| Metric | Result |
|---|---|
| Model Accuracy | **79.8%** |
| True Positives (At-Risk Identified) | **211 customers** |
| Monthly Recurring Revenue (MRR) at Risk | **$16,669** |
| Recall (Churn class) | 65% |

---

## 💰 Business Impact

```
211 at-risk customers × $79/month (churner median) = $16,669/month at risk

At a conservative 30% save rate with proactive intervention:
$16,669 × 30% = ~$5,000/month protected → $60,000/year in preserved revenue
```

This represents a structural shift in Customer Lifetime Value (CLV) management — from reactive, expensive win-back campaigns to proactive, data-driven retention.

---

## 🛠 Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.11+ | Core language |
| Pandas | 2.x | Data manipulation and cleaning |
| Scikit-learn | 1.x | Feature engineering, modeling, evaluation |
| Matplotlib | 3.x | Exploratory visualizations |
| Seaborn | 0.x | Statistical visualizations |
| Jupyter Notebook | — | Interactive development environment |

---

## 📁 Project Structure

```
customer-churn-analysis/
│
├── notebooks/
│   └── analysis.ipynb          # Full analysis — EDA → modeling → results
│
├── src/
│   └── churn_analysis.py       # Modular Python script version
│
├── images/
│   └── telco_churn.png         # Portfolio infographic
│
├── requirements.txt            # Python dependencies
└── README.md
```

---

## ▶️ How to Run

```bash
# 1. Clone the repository
git clone https://github.com/wgalante/customer-churn-analysis
cd customer-churn-analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the notebook
jupyter notebook notebooks/analysis.ipynb
```

---

## 👤 Author

**William Galante**
Data Analyst | Python · SQL · BigQuery · Power BI | AI Agents & Prompt Engineering

- 🌐 [torkai.com.br](https://torkai.com.br)
- 💼 [linkedin.com/in/william-galante](https://linkedin.com/in/william-galante)
- 🐙 [github.com/wgalante](https://github.com/wgalante)
