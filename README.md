# 🎬 Movie Dataset Analysis — Linear & Multiple Regression Models

This analysis explores how movie **revenue** and **ratings** have evolved over time, using **linear regression** for trend discovery and **multiple linear regression** to identify key factors driving box-office performance.  
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

**Visualization**

![Movie Revenue by Year](year_vs_revenue_analysis.png)

### 🔍 Interpretation
The trend line shows a **strong upward correlation** between movie release year and revenue.  
Each passing year corresponds, on average, to an increase of nearly **$2 million** in box-office earnings.

This steady increase can be explained by multiple real-world factors:
- **Inflation and ticket price increases** have naturally raised revenue totals over time.  
- **Global distribution** means films are now released simultaneously across dozens of countries.  
- **Blockbuster franchises** (Marvel, Star Wars, etc.) dominate the market, pulling in massive profits.  
- **Digital marketing and streaming** have extended revenue streams far beyond theater sales.  

The negative intercept (−$3.75B) is only a mathematical artifact — it’s not meaningful, as no films existed near year 0.

### 💡 What This Shows
> The steady rise in revenue reflects the **industrial and commercial growth** of cinema.  
> Movies have evolved from national entertainment into a **multi-billion-dollar global industry**, heavily influenced by technological, economic, and cultural globalization.

---

## ⭐ 2. Average Vote Over Time — Linear Regression

**Model Summary**

- **Slope:** −0.017873 → ratings decrease slightly per year  
- **Intercept:** 42.1072  
- **R²:** 0.0629 (explains 6.29% of variance)  
- **Equation:**

<p align="center">
  <strong><span style="font-size:1.3em;">Average Vote = −0.017873 × Year + 42.1072</span></strong>
</p>

**Visualization**

![Movie Average Vote by Year](year_vs_average_vote_analysis.png)

### 🔍 Interpretation
The red trend line indicates a **small but consistent decline** in movie ratings as years progress.  
However, the R² value (0.06) shows that *year alone is a weak predictor* of rating changes.

This phenomenon can be explained by cultural and social dynamics:
- **Survivorship bias:** only the best older films remain well-known and rated, inflating their averages.  
- **Audience diversity:** millions of modern users contribute ratings, increasing both volume and variance.  
- **Genre saturation:** the explosion of film production leads to a wider range of quality and opinion.  
- **Cultural nostalgia:** older movies often gain “classic” status and higher retrospective acclaim.

### 💡 What This Shows
> Ratings have declined slightly, not because films are worse, but because **the audience and rating ecosystem have evolved**.  
> The internet era democratized criticism — everyone can now rate movies — resulting in more variation and less uniform praise.

---

## 💰 3. Multi-Feature Revenue Prediction — Multiple Linear Regression

**Target:** `revenue_log`  
**Features:** `budget_log`, `runtime`, `popularity_log`, `vote_count_log`, `release_year`  
**Samples:** 3068 (Train = 2454  |  Test = 614)

### Model Performance (Log-space)

| Metric | Value | Interpretation |
|:--------|:------|:---------------|
| **R²** | 0.4620 | Explains 46% of the variance in log-revenue |
| **MSE** | 2.0054 | Average squared log error |
| **MAE** | 0.8853 | Average absolute log error (~$70M in real terms) |

### Feature Importance

| Feature | Coefficient | Interpretation |
|:----------|:-------------|:---------------|
| **vote_count_log** | **0.783** | Strongest predictor — higher engagement drives higher revenue |
| **budget_log** | 0.146 | Bigger budgets produce larger box-office returns |
| **popularity_log** | 0.098 | Social/media buzz contributes to revenue growth |
| **release_year** | −0.0076 | Small negative effect once other variables are controlled |
| **runtime** | 0.0037 | Minimal impact; longer films slightly correlate with higher earnings |

**Visualization**

![Feature Coefficients (log-revenue model)](mlr_revenue_coefficients.png)

### 🔍 Interpretation
This model integrates multiple variables simultaneously to predict revenue (log-transformed to handle extreme values).  
The **R² score of 0.46** indicates a moderate fit — nearly half of revenue variability is explained by these five predictors.

Key takeaways:
- **Audience engagement (vote count)** is the **strongest factor** affecting revenue. The more people watch and rate a movie, the higher its box-office income.  
- **Budget** and **popularity** amplify this — higher investments and marketing visibility boost exposure.  
- **Year** and **runtime** add minimal predictive power, meaning the *time of release* itself matters less than *how well the film is promoted and received*.

### 💡 What This Shows
> Financial success in film today is driven by **engagement and visibility**, not chronology or runtime.  
> High-budget, highly-marketed films with large audiences perform best — reflecting a shift from artistic value to **mass reach and profitability**.

---

## 🎞️ Combined Insights & Broader Meaning

Across all models, we can see distinct but connected trends:

| Observation | Underlying Cause | Interpretation |
|:-------------|:-----------------|:----------------|
| 📈 **Revenue increases** | Inflation, global distribution, franchise culture | The film industry has matured into a large-scale, profit-driven enterprise |
| ⭐ **Ratings decline slightly** | More films, larger audiences, rating democratization | Broader participation leads to more diverse and critical scoring |
| 💬 **Engagement predicts success** | Vote count and popularity dominate | Success is now tied to online visibility and community engagement |
| 🧭 **Year becomes less relevant** | Digital access to all eras of media | The cultural “when” matters less — what matters is reach and resonance |

### 🧠 Final Reflection
> The data illustrates the **evolution of cinema** from a localized art form into a **globalized entertainment industry**.  
> Financial trends mirror economic growth, while audience behavior reflects the shift to participatory, data-driven media culture.  
> The relationship between votes, budget, and revenue highlights how **success has become measurable, predictable, and scalable**, centered on audience size rather than critical acclaim.

---

## 🧩 Skills Demonstrated
- Linear and multiple regression modeling  
- Trend and variance interpretation (R², MSE, MAE)  
- Feature-importance visualization  
- Logarithmic transformation for skewed financial data  
- Critical contextual analysis linking data to real-world factors  

---

**End of Analysis**
