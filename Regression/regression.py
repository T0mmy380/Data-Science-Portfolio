import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline, make_pipeline

# Load the movie dataset
df = pd.read_csv('movie_dataset.csv')

# Linear Regression: Year vs Revenue
def analyze_year_vs_revenue():
    """
    - Analyze the relationship between release year and movie revenue -
    
    Model slope:     1937245.5165748552
    Model slope:     $1.94M
    
    Model intercept: -3751436514.6382027
    Model intercept: -$3.75B


    - Movie revenues increase by $1.94M per year on average
    - The trend line suggests movies at year 0 would have -$3.75B revenue
    
    Y = 1,937,245.52*Year −3,751,436,514.64
    """
    # Extract year from release_date
    df['release_year'] = pd.to_datetime(df['release_date']).dt.year
    
    # Clean data - remove movies with missing revenues or years
    df_clean = df[(df['revenue'] > 0) & (df['release_year'].notna())].copy()
    
    # Filter for movies with reasonable number of votes (to ensure quality ratings)
    df_clean = df_clean[df_clean['vote_count'] >= 50]
    
    
    plt.figure(figsize=(12, 6))
    plt.scatter(df_clean['release_year'], df_clean['revenue'], alpha=0.6, s=30)
    plt.title('Movie Revenue by Year')
    plt.xlabel('Year')
    plt.ylabel('Average Revenue')
    plt.grid(True, alpha=0.3)
    
    # Add a linear regression line
    X = df_clean['release_year'].values.reshape(-1, 1)
    y = df_clean['revenue'].values
    
    # Fit the model
    lr = LinearRegression()
    lr.fit(X, y)
    
    # Generate line points
    line_x = np.linspace(df_clean['release_year'].min(), df_clean['release_year'].max(), 100)
    line_y = lr.predict(line_x.reshape(-1, 1))
    
    # Plot the line
    plt.plot(line_x, line_y, color='red', linewidth=2, label='Trend Line')
    plt.legend()
    
    # Custom formatter for revenue values
    def format_revenue(x, p):
        if x >= 1e9:
            return f'${x/1e9:.1f}B'
        elif x >= 1e6:
            return f'${x/1e6:.0f}M'
        else:
            return f'${x:.0f}'
    
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(format_revenue))
    
    plt.savefig('Regression/year_vs_revenue_analysis.png')
    
    plt.show()
    
    # Calculate R² score
    y_pred = lr.predict(X)
    r2 = r2_score(y, y_pred)
    
    
    # Format and display the model parameters
    def format_revenue_value(value):
        abs_value = abs(value)
        sign = '-' if value < 0 else ''
        
        if abs_value >= 1e9:
            return f'{sign}${abs_value/1e9:.2f}B'
        elif abs_value >= 1e6:
            return f'{sign}${abs_value/1e6:.2f}M'
        elif abs_value >= 1e3:
            return f'{sign}${abs_value/1e3:.2f}K'
        else:
            return f'{sign}${abs_value:.2f}'

    slope = lr.coef_[0]
    intercept = lr.intercept_
    
    print("\n=== LINEAR REGRESSION: YEAR vs REVENUE ANALYSIS ===")
    print(f"Model Slope:     {slope:,.2f} (${format_revenue_value(slope)})")
    print(f"Model Intercept: {intercept:,.2f} ({format_revenue_value(intercept)})")
    print(f"R² Score:        {r2:.4f} (explains {r2*100:.2f}% of variance)")
    print("\nInterpretation:")
    print(f"• Movie revenues increase by {format_revenue_value(slope)} per year on average")
    print(f"• The trend line suggests movies at year 0 would have {format_revenue_value(intercept)} revenue")
    print(f"• Linear equation: Y = {slope:,.2f} * X + ({intercept:,.2f})")
    
# Linear Regression: Year vs Average Vote
def analyze_year_vs_average_vote():
    """
    Analyze the relationship between release year and movie average vote ratings
    """
    # Extract year from release_date
    df['release_year'] = pd.to_datetime(df['release_date']).dt.year
    
    # Clean data - remove movies with missing vote_average or years
    df_clean = df[(df['vote_average'] > 0) & (df['release_year'].notna())].copy()
    
    # Filter for movies with reasonable number of votes (to ensure quality ratings)
    df_clean = df_clean[df_clean['vote_count'] >= 50]
    
    plt.figure(figsize=(12, 6))
    plt.scatter(df_clean['release_year'], df_clean['vote_average'], alpha=0.6, s=30)
    plt.title('Movie Average Vote by Year')
    plt.xlabel('Year')
    plt.ylabel('Average Vote Rating')
    plt.grid(True, alpha=0.3)
    
    # Add a linear regression line
    X = df_clean['release_year'].values.reshape(-1, 1)
    y = df_clean['vote_average'].values
    
    # Fit the model
    lr = LinearRegression()
    lr.fit(X, y)
    
    # Generate line points
    line_x = np.linspace(df_clean['release_year'].min(), df_clean['release_year'].max(), 100)
    line_y = lr.predict(line_x.reshape(-1, 1))
    
    # Plot the line
    plt.plot(line_x, line_y, color='red', linewidth=2, label='Trend Line')
    plt.legend()
    
    # Set y-axis limits to show vote range more clearly
    plt.ylim(0, 10)
    
    plt.savefig('Regression/year_vs_average_vote_analysis.png')
    plt.show()
    
    # Display model parameters
    slope = lr.coef_[0]
    intercept = lr.intercept_
    
    # Calculate R² score
    y_pred = lr.predict(X)
    r2 = r2_score(y, y_pred)
    
    print("\n=== LINEAR REGRESSION: YEAR vs AVERAGE VOTE ANALYSIS ===")
    print(f"Model Slope:     {slope:.6f}")
    print(f"Model Intercept: {intercept:.4f}")
    print(f"R² Score:        {r2:.4f} (explains {r2*100:.2f}% of variance)")
    print("\nInterpretation:")
    print(f"• Movie average votes change by {slope:.4f} points per year on average")
    print(f"• The trend line suggests movies at year 0 would have {intercept:.2f} average vote")
    print(f"• Linear equation: Y = {slope:.6f} * X + {intercept:.4f}")

# Multi-feature Linear Regression: Predict Revenue
def analyze_revenue_multifeature():
    """
    Multiple Linear Regression using: budget, runtime, popularity, vote_count, release_year
    - log1p transform of revenue and skewed inputs for stability
    - train/test split with metrics
    - coefficient bar chart
    """

    # 1) Create release_year and select features
    df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
    features = ['budget', 'runtime', 'popularity', 'vote_count', 'release_year']

    # 2) Filter valid data
    keep = (df['revenue'] > 0) & df['release_year'].notna() & (df['vote_count'] >= 50)
    data = df.loc[keep, features + ['revenue']].copy()

    # 3) Log transforms to handle skewness
    data['revenue_log'] = np.log1p(data['revenue'])
    data['budget_log'] = np.log1p(data['budget'].clip(lower=0))
    data['popularity_log'] = np.log1p(data['popularity'].clip(lower=0))
    data['vote_count_log'] = np.log1p(data['vote_count'].clip(lower=0))

    # 4) Define X and y
    X = pd.DataFrame({
        'budget_log': data['budget_log'],
        'runtime': data['runtime'].fillna(0),
        'popularity_log': data['popularity_log'],
        'vote_count_log': data['vote_count_log'],
        'release_year': data['release_year']
    })
    y = data['revenue_log']

    # 5) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 6) Fit model
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # 7) Predict and evaluate (log-space)
    y_pred_log = lr.predict(X_test)
    r2 = r2_score(y_test, y_pred_log)
    mse = mean_squared_error(y_test, y_pred_log)
    mae = mean_absolute_error(y_test, y_pred_log)

    # 8) Back-transform to original scale (approx. dollars)
    y_test_dollar = np.expm1(y_test)
    y_pred_dollar = np.expm1(y_pred_log)
    mse_dollar = mean_squared_error(y_test_dollar, y_pred_dollar)
    mae_dollar = mean_absolute_error(y_test_dollar, y_pred_dollar)

    # 9) Helper for pretty formatting
    def money(x):
        x = float(x)
        if abs(x) >= 1e9:
            return f"${x/1e9:.2f}B"
        elif abs(x) >= 1e6:
            return f"${x/1e6:.2f}M"
        elif abs(x) >= 1e3:
            return f"${x/1e3:.2f}K"
        return f"${x:.2f}"

    # 10) Print results
    print("\n--------------------------------------------------")
    print("=== MULTIPLE LINEAR REGRESSION REPORT ===")
    print(f"Target: revenue_log | Features: {list(X.columns)}")
    print(f"Samples: {len(X)} | Train: {len(X_train)} | Test: {len(X_test)}")

    print("\nLog-space metrics (model was trained in log-space):")
    print(f"  R²:  {r2:.4f}")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")

    print("\nBack-transformed metrics (approx. in dollars):")
    print(f"  MSE: {money(mse_dollar)}")
    print(f"  MAE: {money(mae_dollar)}")

    # 11) Coefficients
    coef_df = pd.DataFrame({
        'feature': X.columns,
        'coefficient': lr.coef_
    }).sort_values('coefficient', key=abs, ascending=False)

    print("\nTop coefficients:")
    print(coef_df.to_string(index=False))

    # 12) Plot coefficient magnitudes
    plt.figure(figsize=(10, 5))
    plt.barh(coef_df['feature'][::-1], coef_df['coefficient'][::-1])
    plt.title('Feature Coefficients (log-revenue model)')
    plt.xlabel('Coefficient (impact on log(revenue))')
    plt.tight_layout()
    plt.savefig('Regression/mlr_revenue_coefficients.png')
    plt.show()
    
# ====================================================================
# MAIN ANALYSIS EXECUTION
# ====================================================================

print("\n" + "="*70)
print("MOVIE DATASET ANALYSIS - LINEAR & POLYNOMIAL REGRESSION")
print("="*70)


# Revenue Analysis
print("\n" + "-"*50)
print("REVENUE ANALYSIS")
print("-"*50)
analyze_year_vs_revenue()
#analyze_year_vs_revenue_polynomial(degree=2)
#analyze_year_vs_revenue_polynomial(degree=20)

# Average Vote Analysis
print("\n" + "-"*50)
print("AVERAGE VOTE ANALYSIS")
print("-"*50)
analyze_year_vs_average_vote()
#analyze_year_vs_average_vote_polynomial(degree=2)
#analyze_year_vs_average_vote_polynomial(degree=20)

# Multi-feature Revenue Prediction
print("\n" + "-"*50)
print("MULTI-FEATURE REVENUE PREDICTION")
print("-"*50)
analyze_revenue_multifeature()


print("\n" + "="*70)
print("ANALYSIS COMPLETE - Check generated PNG files for visualizations")
print("="*70)
