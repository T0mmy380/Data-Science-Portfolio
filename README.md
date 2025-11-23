# 🎬 Movie Dataset Analysis — Data Modeling Portfolio

This project analyzes movie data to explore trends, relationships, and predictive patterns using multiple types of data models.  
Each section focuses on a specific **modeling category**, such as regression, classification, or clustering, with a consistent format including:  
✔ Purpose  
✔ Model setup  
✔ Visualization  
✔ Interpretation  
✔ Real-world meaning  
✔ Skills demonstrated  

---

# 📊 Section 1: Regression Models

Regression models examine how variables relate to each other and allow prediction of numerical outcomes such as revenue, ratings, and trends over time.

---

## 📈 1.1 Revenue Over Time — Linear Regression

### 🎯 Purpose  
Understand long-term industry revenue trends.

### 📋 Model Summary
- **Slope:** +1,937,245.52  
- **Intercept:** −3,751,436,514.64  
- **Equation:**  
<p align="center"><strong>Revenue = 1,937,245.52 × Year − 3,751,436,514.64</strong></p>

### 📷 Visualization  
<img width="1200" height="600" src="https://github.com/user-attachments/assets/5f877fc9-6ba4-4b5f-9f91-b453f0aacc20" />

### 🔍 Interpretation  
Revenue increases by roughly **$2 million per year**.  
This reveals:
- Growing ticket prices and production budgets  
- International market expansion  
- Technology-supported distribution  
- Rise of high-budget franchises  

### 💡 Meaning  
The model reflects the film industry’s shift into a globalized commercial powerhouse.

---

## ⭐ 1.2 Average Vote Over Time — Linear Regression

### 🎯 Purpose  
Analyze how audience ratings evolve across decades.

### 📋 Model Summary
- **Slope:** −0.017873  
- **Intercept:** 42.1072  
- **R²:** 0.0629  
- **Equation:**  
<p align="center"><strong>Average Vote = −0.017873 × Year + 42.1072</strong></p>

### 📷 Visualization  
<img width="1200" height="600" src="https://github.com/user-attachments/assets/364530db-d50e-4ede-9d3e-6af337841c08" />

### 🔍 Interpretation  
Small downward trend in ratings, but very weak explanatory power.  
Reasons include:
- Survivorship bias  
- Larger, more diverse online audiences  
- Genre oversaturation  
- Shifting cultural norms  

### 💡 Meaning  
Ratings appear lower not because movies worsen, but because rating systems broadened.

---

## 💰 1.3 Multi-Feature Revenue Prediction — Multiple Linear Regression

### 🎯 Purpose  
Predict revenue using several features simultaneously.

### 📋 Key Information
**Target:** `revenue_log`  
**Features:** `budget_log`, `runtime`, `popularity_log`, `vote_count_log`, `release_year`  
**Samples:** 3068 total  

### 📊 Model Performance (Log-space)

| Metric | Value | Meaning |
|-------|-------|---------|
| **R²** | 0.4620 | Explains 46% of revenue variance |
| **MSE** | 2.0054 | Error in log-space |
| **MAE** | 0.8853 | Avg. under/over-prediction |

### 📈 Feature Importance

| Feature | Coefficient | Interpretation |
|---------|-------------|----------------|
| **vote_count_log** | **0.783** | Strongest predictor (audience size) |
| budget_log | 0.146 | Higher budgets → higher returns |
| popularity_log | 0.098 | Moderate effect |
| release_year | −0.0076 | Minimal |
| runtime | 0.0037 | Minimal |

### 📷 Visualization  
<img width="1000" height="500" src="https://github.com/user-attachments/assets/a97d20b4-3a6e-4bab-b080-8707f3b8add1" />

### 🔍 Interpretation  
Audience engagement drives revenue.  
Budget helps, but popularity and runtime matter less.

### 💡 Meaning  
Success depends more on *reach* and *visibility* than traditional production factors.

---

## 🧠 1.4 Combined Insights from Regression

| Observation | Interpretation |
|------------|----------------|
| 📈 Revenue increases | Industry expansion + globalization |
| ⭐ Ratings slightly drop | Broader rating participation |
| 💬 Engagement predicts revenue | Large audiences = financial success |
| 🧭 Year matters less | Streaming breaks “era” boundaries |

---

## 🧩 Skills Demonstrated
- Regression modeling  
- Log-space transformations  
- Feature importance analysis  
- Data interpretation and industry mapping  

---

# 🎯 Section 2: Clustering Models

Clustering identifies natural groups in the data without labels, revealing distinct economic and popularity tiers in the film industry.

---

## 🎥 2.1 Market Segmentation with K-Means

### 🎯 Purpose  
Group movies by spending, revenue, and audience reach.

### 📋 Model Setup

| Step | Detail |
|------|--------|
| Features | budget, revenue, popularity, vote_average |
| Scaling | StandardScaler |
| Model | K-Means (k=3) |
| Evaluation | Elbow Method |

---

## 🧮 Elbow Method

### 📷 Visualization  
<img width="640" height="480" src="https://github.com/user-attachments/assets/f6badb62-0417-46aa-9c15-2a9602e796f7" />

### 🔍 Interpretation  
The “bend” at **k = 3** shows three optimal clusters.

---

## 🎬 Cluster Results (k = 3)

### 📷 Visualization  
<img width="800" height="600" src="https://github.com/user-attachments/assets/70bbf4b4-0f93-4f56-be88-cd19061d5356" />

### 📊 Summary

| Cluster | Avg Budget | Avg Revenue | Popularity | Rating |
|---------|------------|-------------|------------|--------|
| **2 — Blockbusters** | $151M | $641M | 98 | 6.72 |
| **0 — Studio Films** | $54M  | $153M | 38 | 6.52 |
| **1 — Indie Films** | $11M  | $17M  | 9  | 5.88 |

---

## 🔍 Interpretation

### 🎞 Cluster 2 — Blockbusters  
High budget, high revenue, global reach.

### 🍿 Cluster 0 — Studio Films  
Moderate budgets and performance.

### 🎬 Cluster 1 — Indie Films  
Low budgets → limited exposure.

---

## 💡 Meaning  
The film industry operates on a **three-tier economic system**, with budget as the main divider.

---

## 🧩 Skills Demonstrated
- K-Means modeling  
- Elbow Method evaluation  
- Cluster visualization & interpretation  
- Market segmentation analysis  

---

# 🧠 Section 3: Classification Models

Classification models predict categorical outcomes.  
In this section, Logistic Regression is used to classify whether a movie becomes a **Hit (1)** or a **Flop (0)** based on production and audience features.

---

## 🎬 3.1 Movie Success Prediction — Logistic Regression

### 🎯 Purpose  
To determine whether a movie will be a **financial hit** by analyzing measurable features such as budget, vote count, popularity, runtime, and ratings.  
The goal is to understand *what drives success* and *how well the model can separate hits from flops*.

---

## 📋 Model Summary

| Property | Description |
|----------|-------------|
| **Target Variable** | `hit_flag` (1 = Hit, 0 = Flop) |
| **Hit Threshold** | Revenue > \$65,070,412 (median) |
| **Features Used** | `budget_log`, `popularity_log`, `vote_average`, `vote_count_log`, `runtime` |
| **Samples** | 2961 total (Train = 2368 / Test = 593) |
| **Model** | Logistic Regression (max_iter = 1000) |
| **Performance** | Accuracy: 0.816 • Precision: 0.814 • Recall: 0.832 • F1: 0.823 |

---

## 📊 Confusion Matrix — Model Performance

<img width="640" height="480" src="https://github.com/user-attachments/assets/f8fdcabf-5fab-4f2c-b175-e36527c0b2dd" />

### 🔍 Interpretation  
- **484 / 593 correct predictions**  
- Balanced classification between hits and flops  
- **58 flops incorrectly predicted as hits**  
- **51 hits incorrectly predicted as flops**

The model performs strongly and maintains consistency across both classes.  
Most mistakes occur with movies in the “middle zone” (mid-budget / moderate engagement).

### 💡 What This Shows  
> The classifier can reliably distinguish success patterns.  
> Misclassifications are natural for borderline films whose characteristics overlap.

---

## 📈 Feature Probability Curves — Influence of Each Feature

These curves show how a feature changes the predicted probability of a movie being a hit, with all other variables held constant.

---

### 💰 Budget vs Hit Probability

<img width="960" height="600" src="https://github.com/user-attachments/assets/20e869bd-f12d-4479-8cd4-13d558818a0e" />

#### 🔍 Interpretation  
- Low budgets → low hit probability  
- Mid budgets → sharp increase (steep slope)  
- High budgets → plateau near 1.0  

Budget has the **largest impact** on success.

#### 💡 What This Shows  
> Budget drives marketing scale, production value, and distribution reach — making it the most influential predictor.

---

### ⭐ Vote Count vs Hit Probability

<img width="960" height="600" src="https://github.com/user-attachments/assets/c7aa70f1-3b72-4603-81f8-a819eaab63c0" />

#### 🔍 Interpretation  
- Smooth upward curve  
- More votes → higher likelihood of success  
- No dramatic jumps like budget

#### 💡 What This Shows  
> Vote count reflects *audience engagement* — a key factor in achieving strong financial performance.

---

### 🧡 Popularity vs Hit Probability

<img width="960" height="600" src="https://github.com/user-attachments/assets/c1a30e08-89ee-4f4e-8c8f-f83b04454922" />

#### 🔍 Interpretation  
- Very flat curve  
- Popularity alone barely affects hit probability  
- Effect often overshadowed by budget & vote count

#### 💡 What This Shows  
> Popularity is *not* a strong standalone indicator — it reflects temporary hype more than sustained performance.

---

## 🟧 ROC Curve — Overall Classification Ability

<img width="1050" height="1050" src="https://github.com/user-attachments/assets/810eed27-4eaf-4a36-90fa-6907a6cae3f7" />

### 🔍 Interpretation  
- **AUC = 0.906 → Excellent model quality**  
- Curve is close to the top-left corner  
- Model ranks hits higher than flops **90.6% of the time**

### 💡 What This Shows  
> The classifier performs strongly across all possible thresholds — not just at the default 0.5 cutoff.

---

## 🧠 Combined Insights from Classification

| Observation | Interpretation |
|------------|----------------|
| 💰 Budget strongest predictor | Investment → visibility → higher success |
| ⭐ Vote count meaningful | Wider engagement → higher revenue |
| 🧡 Popularity weak | Not a reliable success indicator |
| ⚖ Balanced performance | Good at both hits and flops |
| 🟧 AUC = 0.906 | Strong capability to separate the two classes |

---

## 🧩 Skills Demonstrated in This Section
- Binary classification modeling  
- Confusion matrix analysis  
- Probability curve interpretation  
- ROC-AUC evaluation  
- Understanding feature effects on categorical prediction  
- Connecting predictive patterns to real industry behavior  

---
