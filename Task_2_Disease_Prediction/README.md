# CodeAlpha_DiseasePrediction

**CodeAlpha Machine Learning Internship —  Disease Prediction from Medical Data**
### Advanced level: large-scale, imbalanced real-world dataset

A classification pipeline that predicts heart disease/attack risk from patient health indicators, built with proper ML methodology: validation-based model selection, class-imbalance handling, hyperparameter tuning, probability calibration, and a decision threshold matched to the actual use case rather than a default cutoff.

---

## 🎯 Objective

Predict the possibility of heart disease/attack based on patient health data, as specified in the CodeAlpha ML internship task list.

## 📊 Dataset

**BRFSS 2015 Heart Disease Health Indicators** — 253,680 patients (229,781 after removing duplicates), 21 features, from the CDC's annual Behavioral Risk Factor Surveillance System telephone survey.

This is a **large, real-world, heavily imbalanced** dataset (~10% positive class), chosen deliberately over the smaller, cleaner 303-row UCI Cleveland dataset commonly used for this task — the scale and imbalance force genuinely advanced handling (class weighting, PR-AUC, calibration, threshold tuning) rather than a toy clean split.

Most features are self-reported binary risk indicators or survey categories (`HighBP`, `HighChol`, `Smoker`, `DiffWalk`, etc.) rather than lab values — a population health screening dataset, not clinical lab data. See the notebook for the full feature list and descriptions.

- Train: ~218,000 patients
- Validation: ~10,900 patients
- Test: ~11,500 patients (held out completely, used exactly once)

## 🧠 Approach

Four classifiers compared — **Logistic Regression, Random Forest, SVM (RBF kernel), XGBoost** — with a methodology designed to avoid the common pitfalls of imbalanced medical classification:

1. **Class imbalance handling**: class weighting (`class_weight='balanced'` / `scale_pos_weight`) rather than resampling
2. **Fair, feasible model screening**: all 4 algorithms trained on a stratified 15,000-row subsample (RBF-SVM doesn't scale to 218,000 rows in a free Colab session — this is documented explicitly, not hidden), scored on the full validation set
3. **Model selection on validation PR-AUC** (not accuracy, which is misleading at ~10% positive prevalence; not the test set, to avoid leakage)
4. **Hyperparameter tuning** (`RandomizedSearchCV`) for the winning algorithm
5. **Full-data refit** of the tuned winner on the complete training set
6. **Probability calibration** (`CalibratedClassifierCV`), since class-weighted models' raw probabilities are typically distorted and shouldn't be read as literal risk percentages
7. **Recall-priority threshold selection**: rather than optimizing the decision threshold for F1 (equal weight to precision/recall), the threshold is chosen to maximize precision subject to recall ≥ 75% — because in disease screening, missing a real case (false negative) is costlier than an unnecessary follow-up test (false positive)
8. **Test set touched exactly once**, for final reporting only

## 📈 Results

**Selected model: Logistic Regression** (highest validation PR-AUC among the 4 candidates; tuned hyperparameters: `C=0.01`, `penalty='l2'`, `solver='lbfgs'`)

| Metric | Value |
|---|---|
| Test Accuracy | 74.3% |
| Test Precision | 25.4% |
| **Test Recall** | **76.7%** |
| Test F1-score | 0.382 |
| Test ROC-AUC | 0.831 |
| Test PR-AUC | 0.363 |
| Brier score (calibrated) | 0.078 |

### Why these numbers, and why recall is the headline metric

With only ~10% of patients having heart disease, a model could hit 90% accuracy by predicting "no disease" for everyone — which is why **accuracy is not the metric to optimize here**. ROC-AUC (0.831) and PR-AUC (0.363, vs. a random baseline of ~0.10) show the model has genuine discriminative ability given the ceiling set by noisy, self-reported survey data.

The threshold was deliberately chosen to prioritize **recall (76.7%)** — catching roughly 3 in 4 real disease cases — at the cost of precision (25.4%): about 3 out of 4 flagged patients don't actually have disease. This is an intentional trade-off, not a weakness: in a screening context, a false alarm costs a follow-up test, while a missed case costs a diagnosis. A pure F1-optimized threshold was tested and rejected for this reason — it improved F1 to 0.41 but dropped recall to 53%, which is the wrong trade-off for a screening tool.

### Calibration mattered — a lot

Class weighting (used to handle imbalance) distorts a model's raw output probabilities. Before calibration, the Brier score (lower = better-calibrated probabilities) was **0.169**; after calibration, **0.078** — roughly a 2x improvement in how trustworthy the reported risk percentages are. This means "this patient has a 40% risk" is a meaningfully more honest statement after calibration than before.

## 🚀 Running the Notebook

1. Open `CodeAlpha_DiseasePrediction.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Run all cells top to bottom — total runtime is roughly 3–5 minutes (SVM screening is the single slowest step, at under 2 minutes)
3. The trained model, scaler, and decision threshold save as `disease_prediction_model.joblib`, `disease_prediction_scaler.joblib`, and `disease_prediction_threshold.joblib`

### Predicting on a new patient

Section 13 includes a worked example (`predict_disease_risk()`) showing how to use the saved model + scaler + threshold together for a single new patient record 
## 🛠 Tech Stack

- Python, scikit-learn, XGBoost
- pandas, NumPy, Matplotlib, Seaborn
- joblib (model persistence)

## ⚠️ Important Disclaimer

This model is trained on a large but **self-reported, survey-based** dataset (a telephone survey), not clinical lab data, and is for **educational purposes only**. It is **not a validated diagnostic tool** and must never be used for actual medical decision-making.

Real clinical deployment would require, at minimum: clinically-verified data (not self-report), fairness auditing across demographic subgroups (age, sex, income), external validation on independent populations, regulatory review (e.g. FDA/CE marking for software as a medical device), and use only as a decision-support aid alongside a qualified clinician — never as a standalone diagnosis.

## 📄 About CodeAlpha

This project was completed as part of the [CodeAlpha](https://www.codealpha.tech) Machine Learning Internship program.

