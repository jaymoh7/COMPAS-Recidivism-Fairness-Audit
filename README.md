# COMPAS Recidivism Fairness Audit

This project audits the COMPAS Recidivism dataset for racial bias using IBM's [AI Fairness 360](https://aif360.mybluemix.net/) toolkit.

## 🔍 Objective

Evaluate racial disparities in recidivism prediction and apply mitigation techniques to enhance fairness.

## 📁 Files

- `compas_fairness_audit.py` — Python script for fairness analysis and bias mitigation.
- `compas_fairness_metrics.png` — Bar graph showing fairness metrics before and after mitigation.
- `report.pdf` — Summary of findings and remediation steps. *(To be added)*

## 📦 Tools & Libraries

- Python 3.x
- AI Fairness 360
- Pandas, NumPy, Scikit-learn
- Matplotlib
- Jupyter Notebook or Python environment

## ⚙️ Steps Performed

1. **Dataset Loading**  
   Used `CompasDataset()` from AIF360.

2. **Bias Detection**  
   Measured disparate impact and false positive rates.

3. **Bias Mitigation**  
   Applied Reweighing to rebalance data across racial groups.

4. **Model Training**  
   Trained logistic regression model post-mitigation.

5. **Evaluation**  
   Compared metrics before and after mitigation.

## 📊 Results

- Disparate Impact (original): ~0.6  
- Disparate Impact (after): ~0.9  
- False Positive Rate Difference: Significantly reduced

## 📌 Notes

- This project is for educational/fairness research.
- Further bias mitigation methods may be necessary depending on use case.

## 📎 License

MIT License
