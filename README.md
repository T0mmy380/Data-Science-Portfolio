# 🎬 Movie Dataset Analysis — Linear & Multiple Regression Models

This analysis explores trends in movie **revenue** and **ratings** over time using **linear regression**, and then expands to a **multiple linear regression** model to predict revenue using several features.  

The dataset includes over 3,000 movies, with key variables such as *year of release*, *budget*, *runtime*, *popularity*, *vote count*, and *revenue*.

---

## 📈 1. Revenue Over Time — Linear Regression

**Model Summary**

- **Slope:** +1,937,245.52 → revenues increase by about **$1.94M per year**
- **Intercept:** -3,751,436,514.64
- **Equation:**  
  \[
  \text{Revenue} = 1,937,245.52 \times \text{Year} - 3,751,436,514.64
  \]

**Interpretation**
- The model shows a **strong upward trend**: newer movies consistently earn more revenue.
- This growth reflects *industry expansion, inflation, international markets, and large-scale productions*.
- The negative intercept simply represents extrapolation before data exists (not meaningful in real terms).

**Visualization**

![Movie Revenue by Year](<img width="1200" height="600" alt="year_vs_average_vote_analysis" src="https://github.com/user-attachments/assets/d747df34-b76b-45e7-8c4d-adee7371595f" />)

**Conclusion**
> Revenues have grown significantly over time, confirming long-term financial growth in the film industry.

---

## ⭐ 2. Average Vote Over Time — Linear Regression

**Model Summary**

- **Slope:** -0.017873 → average ratings drop slightly each year.
- **Intercept:** 42.1072  
- **R²:** 0.0629 (explains ~6.3% of variance)

**Equation:**  
\[
\text{Average Vote} = -0.017873 \times \text{Year} + 42.1072
\]

**Interpretation**
- Ratings show a **small downward trend** over the years.
- However, the low R² indicates **year is not a strong predictor** — other factors affect ratings far more.
- The decrease may reflect **wider audience diversity**, *more movie releases*, and *changing rating standards*.

**Visualization**

![Movie Average Vote by Year](year_vs_average_vote_analysis.png)

**Conclusion**
> Average movie ratings have slightly declined over time, but the effect of year is weak.  
> Audience perception, trends, and genre diversity likely have greater influence than release date alone.

---

## 💰 3. Multi-Feature Revenue Prediction — Multiple Linear Regression

**Target:** `revenue_log`  
**Features:** `budget_log`, `runtime`, `popularity_log`, `vote_count_log`, `release_year`  
**Samples:** 3068 (Train = 2454, Test = 614)

### Model Performance (Log-space)
| Metric | Value | Interpretation |
|--------|--------|----------------|
| **R²** | 0.4620 | Explains 46% of revenue variance |
| **MSE** | 2.0054 | Mean Squared Error (log-scale) |
| **MAE** | 0.8853 | Average prediction error in log-scale |

### Back-transformed Metrics (Approximate Dollar Values)
| Metric | Value |
|--------|--------|
| **MSE:** | $20.55B |
| **MAE:** | $69.82M |

---

### Feature Importance

| Feature | Coefficient | Interpretation |
|----------|--------------|----------------|
| **vote_count_log** | 0.783 | Strongest predictor — higher audience engagement drives revenue |
| **budget_log** | 0.146 | Larger budgets modestly increase earnings |
| **popularity_log** | 0.098 | More online/social buzz correlates with revenue |
| **release_year** | -0.0076 | Slight negative effect once other factors are considered |
| **runtime** | 0.0037 | Minimal influence; longer films earn slightly more |

**Visualization**

![Feature Coefficients (log-revenue model)](mlr_revenue_coefficients.png)

---

### Interpretation

- The **vote_count_log** variable dominates the model, showing that **public engagement** is the best predictor of box office success.
- **Budget** and **popularity** also have notable impacts, indicating that *investment and hype* contribute to financial performance.
- **Year** and **runtime** have minimal influence once other variables are included, suggesting that the **modern success of films is driven by audience scale rather than time period.**
- The model’s R² of 0.46 indicates a **moderate fit**, capturing roughly half of the variation in movie revenue.

---

### Summary Conclusion

> The regression analyses demonstrate clear financial growth in cinema over time and reveal that a movie’s success is most strongly tied to its **audience reach and production budget**.  
> While average ratings have shown a mild decline, overall revenue and engagement metrics indicate a thriving and increasingly large-scale movie industry.

---

**Skills Demonstrated**
- Linear regression trend analysis  
- Multi-feature regression modeling  
- Feature importance interpretation  
- Visualization and statistical evaluation (R², MSE, MAE)  
- Log-transformation for skewed data

---

**End of Analysis**
