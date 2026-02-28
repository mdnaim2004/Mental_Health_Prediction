```
███╗   ███╗███████╗███╗   ██╗████████╗ █████╗ ██╗         ██╗  ██╗███████╗ █████╗ ██╗   ████████╗██╗  ██╗
████╗ ████║██╔════╝████╗  ██║╚══██╔══╝██╔══██╗██║         ██║  ██║██╔════╝██╔══██╗██║   ╚══██╔══╝██║  ██║
██╔████╔██║█████╗  ██╔██╗ ██║   ██║   ███████║██║         ███████║█████╗  ███████║██║      ██║   ███████║
██║╚██╔╝██║██╔══╝  ██║╚██╗██║   ██║   ██╔══██║██║         ██╔══██║██╔══╝  ██╔══██║██║      ██║   ██╔══██║
██║ ╚═╝ ██║███████╗██║ ╚████║   ██║   ██║  ██║███████╗    ██║  ██║███████╗██║  ██║███████╗ ██║   ██║  ██║
╚═╝     ╚═╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝╚══════╝    ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚══════╝ ╚═╝   ╚═╝  ╚═╝
```

# 🧠 Prediction of Mental Health Treatment Patterns

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter" />
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-green?style=for-the-badge&logo=scikit-learn" />
  <img src="https://img.shields.io/badge/Kaggle-Notebook-20BEFF?style=for-the-badge&logo=kaggle" />
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge" />
</p>

---

## 📌 Project Overview

This project analyzes and predicts **mental health treatment-seeking behavior** among tech industry employees using real-world survey data. The study combines thorough **data preprocessing**, rich **exploratory data analysis (EDA)**, advanced **feature selection techniques**, and machine learning modeling to uncover what drives people to seek mental health treatment.

The insights from this project can support **healthcare providers**, **HR teams**, and **policymakers** in making data-informed decisions around mental health support in the workplace.

🔗 **Kaggle Notebook:** [Mental Health EDA & Important Features](https://www.kaggle.com/code/mdnaimislam165436/mental-health-eda-important-features)

---

## 📂 Dataset

- **Source:** [Mental Health in Tech Survey](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey) — OSMI (Open Sourcing Mental Illness)
- **File:** `survey.csv`
- **Description:** Survey responses from tech workers worldwide about their mental health conditions, workplace environment, and whether they have sought professional treatment.

---

## 🗂️ Project Structure

```
mental-health-prediction/
│
├── mental-health-prediction.ipynb   # Main Jupyter Notebook
└── README.md                        # Project documentation
```

---

## 🔄 Workflow / Pipeline

The notebook follows a clean, end-to-end machine learning pipeline:

```
Data Loading  ──►  Data Cleaning  ──►  EDA  ──►  Feature Selection  ──►  Model Building  ──►  Evaluation
```

---

## 📋 Step-by-Step Breakdown

### 1️⃣ Setup and Data Loading

All necessary libraries are installed and imported at the beginning of the notebook:

- **Data Manipulation:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Machine Learning:** `scikit-learn`
- **Feature Selection:** `category_encoders`, `skrebate`, `boruta`

The dataset is loaded directly from Kaggle's input path and initially explored using `.columns` and `.head()` to understand its structure.

---

### 2️⃣ Data Cleaning and Preprocessing

Raw survey data is inherently messy. This phase focused on making the data clean, consistent, and model-ready.

**Column Formatting:**
Column names were stripped of whitespace, converted to lowercase, and spaces replaced with underscores for clean access.

**Gender Standardization:**
The gender column had many inconsistent entries like `"maile"`, `"cis male"`, `"femail"`, `"woman"`, etc. All variants were mapped to three clean categories: `male`, `female`, and `other`.

**Age Outlier Removal:**
Unrealistic age values were removed by keeping only respondents aged **18 to 65**. Infinite and NaN values were also handled. An **Age Distribution Histogram** (with KDE curve) was plotted to visualize the resulting distribution.

**Categorical Encoding:**
All categorical/object-type columns were encoded into numerical values using `LabelEncoder`, creating a fully numeric dataset (`data_encoding`) ready for machine learning.

---

### 3️⃣ Exploratory Data Analysis (EDA)

EDA was performed to reveal underlying patterns and relationships within the dataset. The following visualizations were created:

| Visualization | Purpose |
|---|---|
| 📊 **Top 10 Countries (Bar Chart)** | Shows where survey respondents are from |
| 📊 **Top 10 States (Bar Chart)** | Shows state-level distribution of respondents |
| 🥧 **Country Pie Chart** | Proportion of top 5 contributing countries |
| 🥧 **Treatment Distribution Pie Chart** | Overall split between those who sought treatment vs. those who did not |
| 🥧 **State Pie Chart** | Regional treatment patterns across US states |
| 🎻 **Violin Plot — Age vs Treatment by Gender** | Explores how age and gender jointly relate to treatment-seeking behavior |
| 🌡️ **Spearman Correlation Heatmap** | Identifies linear and monotonic relationships between all features and the target |

**Top 10 Features Correlated with Treatment (Spearman):**

> `family_history`, `care_options`, `benefits`, `obs_consequence`, `anonymity`, `work_interfere`, `state`, `country`, `comments`, `mental_health_interview`

These features showed the strongest statistical relationship with whether a person seeks mental health treatment.

---

### 4️⃣ Feature Selection

Multiple feature selection techniques were applied to validate and identify the most predictive features, ensuring the model focuses on only the most relevant variables:

#### ✅ Chi-Square Test
Features were scaled between 0–1 using `MinMaxScaler`, then evaluated using the Chi-Square formula:

$$\chi^2 = \sum \frac{(O - E)^2}{E}$$

Features with the highest scores were selected, indicating a strong statistical dependence with the target variable.

#### ✅ Recursive Feature Elimination (RFE)
`RFE` was applied using `LogisticRegression` as the base estimator. It iteratively removes the least important features until the top 10 are retained. Final selected features:

> `family_history`, `care_options`, `benefits`, `obs_consequence`, `anonymity`, `work_interfere`, `state`, `country`, `comments`, `mental_health_interview`

#### ✅ SMLR (Sparse Multinomial Logistic Regression)
`LogisticRegressionCV` with L2 regularization (5-fold cross-validation) was used to compute feature coefficients. The top 10 features by absolute coefficient magnitude were extracted, showing which features contribute most to the model's decision boundary.

#### ✅ ReliefF (via Mutual Information)
`mutual_info_classif` was used as a proxy for the ReliefF algorithm, scoring features by their ability to distinguish between class labels based on neighboring instances.

#### 📊 Feature Selection Summary Plot
All four techniques were combined into a single DataFrame. A **Feature Counts Bar Chart** was generated to show how many selection methods agreed on each feature — features selected by more methods are considered more important and reliable.

---

### 5️⃣ Model Building

After feature selection, a **Logistic Regression** model was trained using the following top features:

> `work_interfere`, `age`, `family_history`, `care_options`, `state`, `country`, `no_employees`, `leave`, `benefits`, `phys_health_interview`

**Steps followed:**
- Features scaled using `StandardScaler`
- Data split into **training (67%)** and **testing (33%)** sets using `train_test_split` with `random_state=42`
- `LogisticRegression` trained with `max_iter=1000` for convergence

---

### 6️⃣ Model Evaluation

The trained model was evaluated using multiple metrics:

#### 🎯 Accuracy
```
Logistic Regression Accuracy: ~70.46%
```
The model correctly predicts mental health treatment status in approximately **7 out of 10 cases**.

#### 📋 Classification Report

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| 0 (No Treatment) | 0.65 | 0.74 | ~0.70 |
| 1 (Treatment) | 0.76 | 0.67 | ~0.71 |

The model is fairly **balanced across both classes**, with similar F1-scores showing it doesn't strongly favor predicting one class over the other.

#### 🔲 Confusion Matrix
A `ConfusionMatrixDisplay` was plotted to show True Positives, True Negatives, False Positives, and False Negatives — helping identify where the model makes correct vs. incorrect predictions.

#### 📈 ROC Curve & AUC Score
The **Receiver Operating Characteristic (ROC)** curve was plotted by varying the decision threshold. The **AUC (Area Under the Curve)** score quantifies overall classification quality — a value closer to **1.0 is ideal**.

---

## 🛠️ Libraries & Tools Used

| Library | Usage |
|---|---|
| `pandas` | Data manipulation and analysis |
| `numpy` | Numerical computations |
| `matplotlib` | Plotting and visualization |
| `seaborn` | Statistical visualizations |
| `scikit-learn` | ML models, feature selection, evaluation |
| `category_encoders` | Categorical encoding support |
| `skrebate` | ReliefF-based feature selection |
| `boruta` | Boruta feature selection |

---

## ⚙️ How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/mental-health-prediction.git
   cd mental-health-prediction
   ```

2. **Install required libraries:**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn category_encoders skrebate boruta
   ```

3. **Download the dataset** from [Kaggle](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey) and place `survey.csv` in the working directory.

4. **Run the notebook:**
   ```bash
   jupyter notebook mental-health-prediction.ipynb
   ```

> **Note:** If running on Kaggle, the dataset path `/kaggle/input/mental-health-in-tech-survey/survey.csv` is already configured.

---

## 📊 Key Findings

- **Family history** of mental illness is among the strongest predictors of whether someone seeks treatment.
- **Work interference** with mental health significantly impacts treatment-seeking behavior.
- Employees who know about **care options** and **benefits** at their workplace are more likely to seek treatment.
- **Anonymity** concerns appear to play a role in whether individuals seek help.
- The model achieves a solid **~70% accuracy** with balanced precision and recall across both classes.

---

## 🚀 Future Improvements

- Experiment with ensemble models like **Random Forest**, **XGBoost**, or **Gradient Boosting** for improved accuracy
- Apply **SMOTE** for handling class imbalance if present
- Use **SHAP values** for better model interpretability
- Expand feature engineering to extract more meaningful signals from categorical data
- Hyperparameter tuning with **GridSearchCV** or **Optuna**

---

## 👤 Author

**Md. Naim Islam**

<p>
  <a href="https://www.kaggle.com/mdnaimislam165436">
    <img src="https://img.shields.io/badge/Kaggle-Profile-20BEFF?style=for-the-badge&logo=kaggle" />
  </a>
  &nbsp;
  <a href="https://www.linkedin.com/in/md-naim-00a164381/">
    <img src="https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin" />
  </a>
  &nbsp;
  <a href="mailto:naim.cse2004@gmail.com">
    <img src="https://img.shields.io/badge/Email-naim.cse2004@gmail.com-D14836?style=for-the-badge&logo=gmail" />
  </a>
</p>

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">
  ⭐ If you found this project helpful, please consider giving it a star on GitHub!
</p>
