# ml_competition_projects
machine learning projects inspired by competition-style datasets to improve feature engineering, modeling, and data analysis skills for future competitions
-
# Home Credit Risk Predictor
**Live Demo:** https://huggingface.co/spaces/hkamran808/Credit-Default-Risk

Loan default probability model on the Kaggle Home Credit dataset
(307,511 applicants, 120+ features).

**What makes it interesting:**
- Hyperparameter tuning with Optuna (20 trials, maximising ROC-AUC)
- Model stacking: LightGBM base model + Logistic Regression meta-model
  trained on out-of-fold predictions
- Experiment tracking with MLflow — full run history logged
- Feature engineering: credit-to-income ratio, annuity-to-income ratio

**Tech:** LightGBM · scikit-learn · Optuna · MLflow · pandas · NumPy

---

## Skills demonstrated across projects
- Handling class imbalance
- Preventing data leakage (proper train/test splits, OOF validation)
- Hyperparameter search (GridSearchCV, Optuna)
- Model ensembling and stacking
- Production deployment (Streamlit + HuggingFace Spaces)
