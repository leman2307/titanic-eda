# Titanic Survival Analysis — Exploratory Data Analysis

**Author:** Leman  
**Status:** Complete  
**Type:** Exploratory Data Analysis (EDA) & Data Preprocessing  
**Dataset:** [Titanic Passenger Data](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv) — 891 rows, 12 features

---

## Project Overview

This project performs a full exploratory data analysis on the Titanic passenger dataset to identify the key factors that determined survival. The analysis covers data cleaning, feature engineering, visualization, and a baseline model — forming the foundation for a future machine learning classifier.

**Key question:** *Who survived the Titanic, and why?*

---

## Key Findings

| Finding | Detail |
|---|---|
| Overall survival rate | 38.4% of passengers survived |
| Strongest predictor | Sex — women survived at 74% vs men at 19% |
| Class effect | 1st class 63%, 2nd class 47%, 3rd class 24% |
| Interaction effect | Sex and class together predict better than either alone |
| Age signal | Weak — children slightly prioritized but small overall effect |
| Fare correlation | Survivors paid 2.5× more in median fare (correlates with class) |
| Multicollinearity | Pclass and Fare correlate at -0.55 — overlapping information |
| Baseline accuracy | 78.7% — sex-only rule (Phase 2 ML target: beat this) |

---

## Project Structure

```
titanic-eda/
│
├── titanic_eda.ipynb          # Main Jupyter notebook
├── README.md                  # This file
│
└── charts/
    ├── chart1_survival_by_sex.png
    ├── chart2_survival_by_class.png
    ├── chart3_age_distribution.png
    ├── chart4_fare_by_survival.png
    ├── chart5_correlation_heatmap.png
    └── chart6_survival_class_sex.png
```

---

## Methodology

### 1. Data Cleaning
- **Age (20% missing):** Filled with median (resistant to outliers) + added `age_missing` binary flag
- **Cabin (77% missing):** Extracted signal as `cabin_known` binary column — missingness carries information about passenger wealth
- **Embarked (0.2% missing):** Filled with mode (negligible missingness, random)

### 2. Feature Engineering
- **Sex:** Label encoded (female=0, male=1) — binary category
- **Embarked:** One-hot encoded with `pd.get_dummies()` — 3 unordered categories, avoids false ordinality
- **Dropped:** `Name`, `Ticket`, `PassengerId` — identifiers with zero predictive signal

### 3. Exploratory Analysis
Six visualizations answering progressively deeper questions:
1. Overall survival distribution
2. Survival rate by sex
3. Survival rate by passenger class
4. Age distribution — survivors vs non-survivors
5. Fare distribution by survival outcome
6. Survival by class AND sex combined (interaction effect)

### 4. Baseline Model
A one-rule classifier: *predict survived if female, predict died if male.*
This achieves **78.7% accuracy** — the benchmark for Phase 2 ML models.

---

## Visualizations

### Survival by Sex
Women survived at 74%, men at 19% — the strongest single predictor in the dataset.

### Survival by Passenger Class
Monotonically decreasing relationship — higher class number, lower survival rate.

### Correlation Heatmap
Strongest correlates with survival: `Pclass` (-0.34), `Fare` (0.26). `PassengerId` (-0.01) confirms it carries no predictive value.

---

## Technologies Used

| Tool | Purpose |
|---|---|
| Python 3 | Core language |
| Pandas | Data loading, cleaning, EDA |
| NumPy | Numerical operations |
| Matplotlib | Chart foundation |
| Seaborn | Statistical visualizations |
| Jupyter Notebook | Interactive analysis environment |

---


## Next Steps

This EDA forms the foundation for Phase 2 of the project:
- Train a Logistic Regression classifier
- Train a Random Forest classifier  
- Apply feature engineering to handle multicollinearity (Pclass/Fare)
- Target: exceed 78.7% baseline accuracy
- Evaluate with precision, recall, ROC-AUC — not just accuracy

---

## What I Learned

- Real datasets are messy — 3 columns had missing values requiring different strategies based on missingness patterns
- One-hot encoding vs label encoding — the choice prevents false mathematical relationships between categories
- EDA before modeling reveals which features actually matter — avoiding wasted computation on irrelevant inputs
- A simple one-rule baseline (78.7%) sets a meaningful benchmark for evaluating ML models
- Multicollinearity between Pclass and Fare (-0.55) means both features tell overlapping stories — a challenge for feature selection in Phase 2