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

*(Next Sections: Classification Models, Clustering Models, Forecasting Models, etc.)*
