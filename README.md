# 🎬 Movie Dataset Analysis — Data Modeling Portfolio

This project analyzes movie data to explore trends, relationships, and predictive patterns using multiple types of data models.  
Each section focuses on a specific **modeling category**, such as regression, classification, or clustering, with its own findings and visual results.

---

## 📊 Section 1: Regression Models

Regression models are used to find relationships between variables and to make numeric predictions.  
In this section, **linear** and **multiple linear regression** models were applied to understand movie trends in **revenue** and **ratings** over time.

---

### 📈 1.1 Revenue Over Time — Linear Regression

**Model Summary**

- **Slope:** +1,937,245.52 → revenues increase by about **$1.94M per year**  
- **Intercept:** −3,751,436,514.64  
- **Equation:**

<p align="center">
  <strong><span style="font-size:1.3em;">Revenue = 1,937,245.52 × Year − 3,751,436,514.64</span></strong>
</p>

**Visualization**

<img width="1200" height="600" alt="year_vs_revenue_analysis" src="https://github.com/user-attachments/assets/5f877fc9-6ba4-4b5f-9f91-b453f0aacc20" />

#### 🔍 Interpretation
The trend line shows a **strong upward correlation** between movie release year and revenue.  
Each passing year corresponds to roughly **$2 million** more in box-office earnings.

This growth reflects the transformation of the movie industry:
- **Inflation & pricing:** ticket prices and production costs increased over time.  
- **Global reach:** international markets now contribute significantly to box-office totals.  
- **Technological growth:** streaming, merchandise, and digital releases expanded revenue sources.  
- **Blockbuster culture:** high-budget franchises dominate modern film revenue.

#### 💡 What This Shows
> The consistent upward trend demonstrates the **industrial and commercial evolution** of film, reflecting both economic inflation and the shift toward globalized entertainment.

---

### ⭐ 1.2 Average Vote Over Time — Linear Regression

**Model Summary**

- **Slope:** −0.017873 → ratings decrease slightly per year  
- **Intercept:** 42.1072  
- **R²:** 0.0629 (explains 6.29% of variance)  
- **Equation:**

<p align="center">
  <strong><span style="font-size:1.3em;">Average Vote = −0.017873 × Year + 42.1072</span></strong>
</p>

**Visualization**

<img width="1200" height="600" alt="year_vs_average_vote_analysis" src="https://github.com/user-attachments/assets/364530db-d50e-4ede-9d3e-6af337841c08" />

#### 🔍 Interpretation
A mild **downward trend** exists in average ratings over time.  
However, the low R² value means **release year alone is not a strong predictor** of a movie’s rating.

Sociocultural reasons behind this pattern:
- **Survivorship bias:** older movies that are still remembered tend to be high-quality “classics.”  
- **Expanded audience:** millions of global users now rate movies, increasing score variation.  
- **Genre saturation:** the rise of mass production dilutes average quality.  
- **Shifting standards:** changing cultural norms and nostalgia influence rating perception.

#### 💡 What This Shows
> The data suggests **no real decline in quality**, but rather a change in how audiences consume and rate movies.  
> The democratization of online ratings has diversified opinions, making averages appear lower.

---

### 💰 1.3 Multi-Feature Revenue Prediction — Multiple Linear Regression

**Target Variable:** `revenue_log`  
**Features:** `budget_log`, `runtime`, `popularity_log`, `vote_count_log`, `release_year`  
**Samples:** 3068 (Train = 2454  |  Test = 614)

#### Model Performance (Log-space)

| Metric | Value | Interpretation |
|:--------|:------|:---------------|
| **R²** | 0.4620 | Explains 46% of variance in log-revenue |
| **MSE** | 2.0054 | Mean Squared Error (log-space) |
| **MAE** | 0.8853 | Avg. prediction error (~$70M) |

#### Feature Importance

| Feature | Coefficient | Interpretation |
|:----------|:-------------|:---------------|
| **vote_count_log** | **0.783** | Strongest predictor — audience size drives revenue |
| **budget_log** | 0.146 | Higher production budgets moderately increase revenue |
| **popularity_log** | 0.098 | Online/social media buzz correlates with sales |
| **release_year** | −0.0076 | Minimal negative effect after adjusting for other factors |
| **runtime** | 0.0037 | Very small positive effect |

**Visualization**

<img width="1000" height="500" alt="mlr_revenue_coefficients" src="https://github.com/user-attachments/assets/a97d20b4-3a6e-4bab-b080-8707f3b8add1" />

#### 🔍 Interpretation
This model predicts revenue based on several features.  
An **R² of 0.46** means it captures nearly half of the variation in movie earnings.

Main findings:
- **Audience engagement (vote count)** is the most critical predictor of financial success.  
- **Budget** and **popularity** enhance revenue potential by increasing exposure.  
- **Year** and **runtime** become statistically insignificant — success now depends on *reach* and *visibility*, not time.

#### 💡 What This Shows
> Revenue is not just a function of time or quality — it’s driven by **audience attention, marketing scale, and engagement**.  
> The model highlights the film industry’s shift toward **data-driven commercial strategy**, where popularity metrics strongly predict financial outcomes.

---

### 🧠 1.4 Combined Insights from Regression Models

| Observation | Likely Cause | Interpretation |
|:-------------|:-------------|:----------------|
| 📈 **Revenue increases** | Inflation, globalization, franchise dominance | Reflects commercial and industrial growth of cinema |
| ⭐ **Ratings decline slightly** | Broader audiences, nostalgia bias, genre saturation | Ratings reflect diversity of taste, not declining quality |
| 💬 **Engagement predicts success** | Vote counts and popularity dominate | Social visibility is the new measure of success |
| 🧭 **Year is less important** | All eras accessible via streaming | “When” a movie is released matters less than “who” it reaches |

#### 🎬 Real-World Meaning
> The regression results illustrate cinema’s evolution from an artistic pursuit to a **globalized, data-driven entertainment market**.  
> Economic growth, cultural expansion, and online engagement shape success more than traditional measures of critical acclaim.

---

## 🧩 Skills Demonstrated in This Section
- Linear and multiple regression modeling  
- Trend and variance analysis (R², MSE, MAE)  
- Logarithmic transformation for skewed data  
- Feature importance visualization and interpretation  
- Connecting statistical results to real-world cultural and economic factors  

---

## 🎯 Section 2: Clustering Models

Clustering models are used to discover **natural groupings** within data without predefined labels.  
Here, **K-Means Clustering** was applied to the movie dataset to identify economic and popularity segments within the film industry.

---

### 🎥 2.1 Market Segmentation with K-Means Clustering

**Goal:**  
To group movies by *budget*, *revenue*, *popularity*, and *vote_average* to reveal hidden patterns such as **blockbusters**, **studio films**, and **independent productions**.

---

### ⚙️ Model Summary

| Step | Description |
|:------|:-------------|
| **Features Used** | `budget`, `revenue`, `popularity`, `vote_average` |
| **Scaling** | Data normalized using `StandardScaler` |
| **Model** | K-Means (n_clusters = 3, random_state = 42) |
| **Evaluation Method** | Elbow Method (Inertia plot) |

---

### 📈 Elbow Method for Optimal K

The **Elbow Method** helps find the ideal number of clusters by analyzing where inertia (within-cluster variance) stops decreasing rapidly.

**Visualization**

<img width="640" height="480" alt="kmeans_elbow_method" src="https://github.com/user-attachments/assets/f6badb62-0417-46aa-9c15-2a9602e796f7" />

#### 🔍 Interpretation
The “elbow” appears at **K = 3**, suggesting that three clusters best represent distinct groups in the data.  
Beyond three, additional clusters add minimal improvement, indicating natural segmentation into **three main movie categories**.

---

### 🎬 Movie Clustering Results (K = 3)

**Visualization**

<img width="800" height="600" alt="kmeans_clusters" src="https://github.com/user-attachments/assets/70bbf4b4-0f93-4f56-be88-cd19061d5356" />

**Cluster Summary**

| Cluster | Avg. Budget ($) | Avg. Revenue ($) | Avg. Popularity | Avg. Rating |
|:--:|:--:|:--:|:--:|:--:|
| **0** | 54,104,710 | 153,146,300 | 38.35 | 6.52 |
| **1** | 11,077,140 | 17,270,430 | 9.77 | 5.88 |
| **2** | 151,637,700 | 641,251,000 | 98.46 | 6.72 |

---

### 🔍 Interpretation of Clusters

#### 🎞️ **Cluster 2 — Blockbusters**
- **Highest budgets and revenues**
- **Most popular and above-average ratings**
- Represents major commercial hits or global franchises (e.g., Marvel, Star Wars)

#### 🍿 **Cluster 0 — Standard Studio Films**
- **Moderate budgets and moderate revenues**
- **Average popularity and ratings**
- Represents typical wide-release films with balanced success

#### 🎬 **Cluster 1 — Indie/Low-Budget Productions**
- **Lowest budgets, lowest revenues**
- **Low popularity and slightly lower ratings**
- Represents smaller, niche, or limited-release projects

---

### 💡 What This Shows
> The K-Means algorithm successfully discovered **three economic tiers** in the movie industry.  
> Budget and revenue are the strongest dividing factors, while popularity correlates strongly with both.  
> Interestingly, **ratings remain fairly consistent**, showing that higher budgets boost exposure — not necessarily quality.

---

### 🧠 Technical Notes
- Clusters were computed using **4D feature space**, but the visualization shows only 2D (Budget vs Revenue).  
- Some overlap in the scatter plot is expected since **true separation occurs in higher dimensions**.  
- Dimensionality reduction (PCA) can be applied for cleaner visualization of high-dimensional clusters.

---

### 📊 Combined Insights from Clustering
| Insight | Explanation |
|:---------|:-------------|
| 🎥 **3 main market segments** | Indie → Studio → Blockbuster |
| 💰 **Budget drives visibility** | Higher investment increases reach and marketing power |
| ⭐ **Ratings remain stable** | Quality perception doesn’t scale directly with spending |
| 🌐 **Popularity linked to exposure** | Commercial success correlates with visibility, not necessarily content quality |

---

### 🧩 Skills Demonstrated in This Section
- Unsupervised machine learning (K-Means)
- Feature scaling and normalization
- Elbow Method for cluster selection
- Multivariate visualization (Budget vs Revenue)
- Cluster interpretation and market segmentation analysis

---

## 🧠 Section 3: Classification Models

Classification models predict **categories** instead of continuous numeric values.  
In this project, a **Logistic Regression** model is used to classify whether a movie is a **Hit (1)** or a **Flop (0)** based on a set of production and audience-related features.

---

# 🎬 3.1 Movie Success Prediction — Logistic Regression

## 🎯 Goal  
Predict whether a movie will be a **financial hit** using measurable production and audience inputs.

---

## 📋 Model Setup

| Property | Description |
|----------|-------------|
| **Target Variable** | `hit_flag` → 1 = Hit, 0 = Flop |
| **Hit Threshold** | Revenue > \$65,070,412 (median revenue) |
| **Features Used** | `budget_log`, `popularity_log`, `vote_average`, `vote_count_log`, `runtime` |
| **Samples** | 2961 total (Train = 2368 / Test = 593) |
| **Model** | Logistic Regression (max_iter = 1000) |
| **Type** | Supervised Binary Classification |

---

## 📈 Model Performance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Accuracy** | 0.816 | ~81.6% of movies correctly classified |
| **Precision** | 0.814 | When the model predicts *Hit*, it is correct ~81% of the time |
| **Recall** | 0.832 | The model catches ~83% of actual hits |
| **F1 Score** | 0.823 | Balanced precision–recall performance |

Overall, the model performs **very well**, especially for a simple linear classifier.

---

# 🔵 Confusion Matrix — Hit vs Flop Prediction

<img width="640" height="480" alt="logistic_confusion_matrix" src="https://github.com/user-attachments/assets/f8fdcabf-5fab-4f2c-b175-e36527c0b2dd" />

| True Label | Predicted Flop (0) | Predicted Hit (1) |
|------------|--------------------|------------------|
| **Actual Flop (0)** | 231 | 58 |
| **Actual Hit (1)** | 51 | 253 |

### 📝 Interpretation

- **484 / 593 correct predictions** (~82% accuracy)
- **58 False Positives:** Flops incorrectly predicted as hits  
- **51 False Negatives:** Hits incorrectly predicted as flops  
- Balanced distribution of errors  
- Slightly higher recall → model focuses more on identifying hits

### 💡 Meaning  
The model does a strong job separating hits and flops, with most errors occurring in borderline cases.

---

# 📈 Feature Probability Curves — How Each Feature Influences Hit Likelihood

These logistic curves display how changing a single feature affects **predicted probability of hit**, with all other features held constant at their median values.

---

## 💰 Budget vs Hit Probability

<img width="960" height="600" alt="logit_curve_budget" src="https://github.com/user-attachments/assets/20e869bd-f12d-4479-8cd4-13d558818a0e" />

### Interpretation

- Classic **S-shaped logistic curve**
- Low budgets → hit probability near 0  
- Mid budgets → *very fast rise* in probability (steep slope)  
- High budgets → hit probability approaches 1  
- Strongest predictor in the model

### ✔ Meaning  
Budget is the **most powerful feature**. Small increases near the mid-range cause large jumps in success likelihood due to factors like marketing, production scale, and distribution.

---

## ⭐ Vote Count vs Hit Probability

<img width="960" height="600" alt="logit_curve_votes" src="https://github.com/user-attachments/assets/c7aa70f1-3b72-4603-81f8-a819eaab63c0" />

### Interpretation

- Smooth logistic curve  
- Higher vote counts correlate with increased hit probability  
- Rise is more gradual than budget  
- Represents audience engagement and visibility

### ✔ Meaning  
Vote count is a **strong but less dramatic predictor**; it steadily increases hit probability as viewer engagement grows.

---

## 🧡 Popularity vs Hit Probability

<img width="960" height="600" alt="logistic_probability_curve" src="https://github.com/user-attachments/assets/c1a30e08-89ee-4f4e-8c8f-f83b04454922" />

### Interpretation

- Nearly flat curve  
- Popularity alone has minimal effect on predicting hits  
- Overlaps heavily with other features such as budget and marketing

### ✔ Meaning  
Popularity is a **weak standalone predictor** of financial success.

---

# 🟧 ROC Curve — Overall Classification Ability

<img width="1050" height="1050" alt="roc_curve" src="https://github.com/user-attachments/assets/810eed27-4eaf-4a36-90fa-6907a6cae3f7" />

### 🧠 What the ROC Curve Shows

- Plots **True Positive Rate (TPR)** vs **False Positive Rate (FPR)**  
- Curves that hug the **top-left corner** indicate excellent performance  
- Your model achieves **AUC = 0.906**

### 📝 Interpretation

- AUC of 0.906 means the model ranks actual hits higher than flops **90.6% of the time**  
- Steep rise at low FPR values → model identifies many true hits without generating many false positives  
- Confirms the model's strong discriminative power

---

# 🔗 How the Curves, Matrix, and ROC Work Together

| Visualization | Purpose | What It Shows |
|---------------|----------|----------------|
| **Probability Curves** | Shows feature impact | How each feature changes hit likelihood |
| **Confusion Matrix** | Evaluates threshold = 0.5 | How many predictions are correct/incorrect |
| **ROC Curve** | Evaluates all thresholds | Model's overall separation ability |

### Summary  
- Feature curves show **how the model thinks**  
- Confusion matrix shows **how correct it is**  
- ROC curve shows **how powerful the model is overall**

---

# 🧩 Combined Insights from Classification

| Observation | Interpretation |
|-------------|----------------|
| 💰 Budget strongly drives success | High investment increases the chance of a hit |
| ⭐ Vote count predicts audience reception | More votes → higher success probability |
| 🧡 Popularity is weak | Not reliable alone |
| ⚖ Balanced precision/recall | Reliable classification for both classes |

---

*(Next Sections: Forecasting Models, Recommendation Systems, etc.)*

