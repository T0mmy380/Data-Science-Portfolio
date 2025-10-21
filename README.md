# 🎬 Movie Dataset Analysis — Linear & Multiple Regression Models

This analysis explores trends in movie **revenue** and **ratings** over time using **linear regression**, and expands to a **multiple linear regression** model predicting revenue based on several contributing features.  
The dataset includes over 3,000 movies with features such as *release year, budget, runtime, popularity, vote count,* and *revenue*.

---

## 📈 1. Revenue Over Time — Linear Regression

**Model Summary**

- **Slope:** +1,937,245.52 → revenues increase by about **$1.94M per year**
- **Intercept:** −3,751,436,514.64  
- **Equation:**

<p align="center">
  <strong><span style="font-size:1.3em;">Revenue = 1,937,245.52 × Year − 3,751,436,514.64</span></strong>
</p>

**Interpretation**

- The model shows a **strong upward trend**: newer movies consistently earn more.
- This increase reflects **industry expansion**, **inflation**, and the rise of **global blockbuster releases**.
- The negative intercept represents an unrealistic extrapolation for year 0 — it’s not meaningful beyond showing model direction.

**Visualization**

![Movie Revenue by Year](year_vs_revenue_analysis.png)

**Conclusion**

> Movie revenues have increased sharply throughout the 20th and 21st centuries, driven by larger production budgets, globalization, and technological advances in distribution and marketing.

---

## ⭐ 2. Average Vote Over Time — Linear Regression

**Model Summary**

- **Slope:** −0.017873 → average ratings decline slightly each year.
- **Intercept:** 42.1072  
- **R²:** 0.0629 (explains 6.29% of variance)
- **Equation:**

<p align="center">
  <strong><span style="font-size:1.3em;">Average Vote = −0.017873 × Year + 42.1072</span></strong>
</p>

**Interpretation**

- There’s a **small downward trend** in movie ratings over time.
- With R² ≈ 0.06, the correlation is weak — **year alone does not explain ratings**.
- The decline may reflect **rating diversity**, **genre saturation**, and **cultural memory bias** favoring older classics.

**Visualization**

![Movie Average Vote by Year](year_vs_average_vote_analysis.png)

**Conclusion**

> Ratings show a mild decline over the decades, suggesting that while more movies are being produced, audiences have become more varied and critical.  
> Older films that remain in memory tend to be the most acclaimed, leading to higher averages in earlier years.

---

## 💰 3. Multi-Feature Revenue Prediction — Multiple Linear Regression

**Target:** `revenue_log`  
**Features:** `budget_log`, `runtime`, `popularity_log`, `vote_count_log`, `release_year`  
**Samples:** 3068 (Train = 2454  |  Test = 614)

### Model Performance (Log-space)

| Metric | Value | Interpretation |
|:--------|:------|:---------------|
| **R²** | 0.4620 | Explains ~46 % of revenue variance |
| **MSE** | 2.0054 | Mean Squared Error (log-space) |
| **MAE** | 0.8853 | Avg. prediction error (log-space) |

### Back-Transformed (Approximate Dollar Values)

| Metric | Value |
|:--------|:------|
| **MSE:** | $20.55 B |
| **MAE:** | $69.82 M |

---

### Feature Importance

| Feature | Coefficient | Interpretation |
|:----------|:-------------|:---------------|
| **vote_count_log** | **0.783** | Strongest predictor — audience engagement directly drives revenue |
| **budget_log** | 0.146 | Bigger budgets yield moderately higher returns |
| **popularity_log** | 0.098 | Social/media buzz contributes to box-office success |
| **release_year** | −0.0076 | Slight negative effect once other factors are considered |
| **runtime** | 0.0037 | Minimal influence; longer films earn slightly more |

**Visualization**

![Feature Coefficients (log-revenue model)](mlr_revenue_coefficients.png)

**Interpretation**

- The **vote count** variable dominates, showing that *how many people engage with a film* best predicts revenue.  
- **Budget** and **popularity** further reinforce that *investment and visibility* drive financial success.  
- **Year** becomes statistically irrelevant when other features are included — success depends more on audience reach than release time.  
- The model’s **R² = 0.46** indicates a **moderate fit**, explaining nearly half the variation in log-revenue.

---

## 🎞️ 4. In-Depth Discussion and Real-World Insights

### **1️⃣ Industry Growth**
Movie revenues rise steadily because of:
- Inflation and higher ticket prices  
- Global distribution and marketing expansion  
- The rise of **franchises** and cinematic universes  
- Digital distribution and merchandise sales  

> The upward revenue trend mirrors the **economic evolution of the entertainment industry** from art to global commerce.

---

### **2️⃣ Democratization of Ratings**
The slight rating decline can be explained by:
- **Survivorship bias:** only great older films remain remembered and rated  
- **Audience diversity:** millions of user reviewers with differing tastes  
- **Genre saturation:** far more films now compete for similar scores  
- **Changing standards:** nostalgia often elevates older works  

> Lower modern averages do not imply worse quality — they reflect a **broader, more diverse reviewing landscape**.

---

### **3️⃣ Drivers of Financial Success**
From the multiple regression:
- **Audience engagement (vote count)** is the key determinant of profit.  
- **Budget** and **popularity** act as amplifiers — more funding and hype create visibility.  
- **Year** and **runtime** add little value; they’re contextual, not causal.

> Today’s film success depends on **scale, exposure, and social conversation**, not just production year or duration.

---

### **4️⃣ Broader Cultural & Economic Context**

| Theme | Data Observation | Real-World Connection |
|:------|:------------------|:----------------------|
| **Economic Expansion** | Continuous revenue growth | Reflects the film industry’s globalization and commercialization |
| **Cultural Saturation** | Slight decline in ratings | More films dilute attention; critics and audiences differ |
| **Data Democratization** | Votes predict success | Online platforms give the audience collective power |
| **Shift in Value** | Popularity > Critique | Studios optimize for marketability over artistry |

---

## 🧠 Final Reflection

> The data illustrates how cinema has transformed from a niche art form into a **data-driven global enterprise**.  
> Revenue growth is propelled by **audience scale and investment**, not necessarily by higher creative quality.  
> Meanwhile, the slight decline in ratings reflects **mass participation and evolving cultural standards** rather than diminishing craftsmanship.  
> In short, **modern film success is built on visibility, reach, and engagement** — a perfect mirror of today’s digital, connected world.

---

## 🧩 Skills Demonstrated
- Linear & multiple regression modeling  
- Trend and variance interpretation (R², MSE, MAE)  
- Log transformation for skewed data  
- Feature-importance visualization  
- Critical data interpretation and contextual analysis  

---

**End of Analysis**
