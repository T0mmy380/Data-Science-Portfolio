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

def analyze_movie_hit_prediction():
    """
    Logistic Regression - Predict whether a movie is a financial HIT (1) or FLOP (0)
    
    Features:
        budget_log, popularity_log, vote_average, vote_count_log, runtime
    Target:
        hit_flag (binary: 1 if revenue > median, else 0)

    Steps:
        1. Clean data (filter missing, log-transform skewed)
        2. Define binary target variable
        3. Train/test split (80/20)
        4. Fit logistic regression model
        5. Evaluate with accuracy, precision, recall, F1
        6. Print classification report
        7. Optional: visualize confusion matrix
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

    # Prepare data (similar to regression)
    df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
    keep = (df['revenue'] > 0) & (df['budget'] > 0) & (df['vote_count'] >= 50)
    data = df.loc[keep, ['budget', 'revenue', 'popularity', 'vote_average', 'vote_count', 'runtime']].copy()

    # Log transform skewed data
    data['budget_log'] = np.log1p(data['budget'])
    data['popularity_log'] = np.log1p(data['popularity'])
    data['vote_count_log'] = np.log1p(data['vote_count'])

    # Define binary target: 1 = hit (revenue above median)
    median_revenue = data['revenue'].median()
    data['hit_flag'] = (data['revenue'] > median_revenue).astype(int)

    # Features and target
    X = data[['budget_log', 'popularity_log', 'vote_average', 'vote_count_log', 'runtime']].fillna(0)
    y = data['hit_flag']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)

    # Predict
    y_pred = log_reg.predict(X_test)

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n--------------------------------------------------")
    print("=== LOGISTIC REGRESSION REPORT: HIT vs FLOP ===")
    print(f"Samples: {len(X)} | Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Revenue median threshold: ${median_revenue:,.0f}")
    print("\nMetrics:")
    print(f"  Accuracy : {acc:.3f}")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall   : {rec:.3f}")
    print(f"  F1-score : {f1:.3f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Flop (0)", "Hit (1)"])
    disp.plot(cmap="Blues")
    plt.title("Logistic Regression Confusion Matrix: Movie Hit Prediction")
    plt.savefig("Classification/logistic_confusion_matrix.png")
    plt.show()
    
def plot_roc_curve(savepath='Classification/roc_curve.png'):
    """
    Train a logistic regression model (same preprocessing used elsewhere),
    compute ROC curve and AUC on the test split, save plot to `savepath`,
    and print the AUC value.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_curve, auc

    # Prepare data (same filtering and transforms as other functions)
    keep = (df['revenue'] > 0) & (df['budget'] > 0) & (df['vote_count'] >= 50)
    data = df.loc[keep, ['budget','revenue','popularity','vote_average','vote_count','runtime']].copy()

    # Log transforms
    data['budget_log'] = np.log1p(data['budget'])
    data['popularity_log'] = np.log1p(data['popularity'])
    data['vote_count_log'] = np.log1p(data['vote_count'])

    # Binary target by median revenue
    rev_med = data['revenue'].median()
    data['hit_flag'] = (data['revenue'] > rev_med).astype(int)

    # Features & label
    feat_cols = ['budget_log','popularity_log','vote_average','vote_count_log','runtime']
    X = data[feat_cols].fillna(0)
    y = data['hit_flag']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train logistic regression
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)

    # Predict probabilities on test set
    y_prob = log_reg.predict_proba(X_test)[:, 1]

    # Compute ROC and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    auc_val = auc(fpr, tpr)

    # Plot
    plt.figure(figsize=(7,7))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_val:.3f})')
    plt.plot([0,1], [0,1], color='navy', lw=1, linestyle='--', label='Chance')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Logistic Regression (Movie Hit Prediction)')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(savepath, dpi=150)
    plt.show()

    print(f"Saved ROC curve to: {savepath} (AUC = {auc_val:.3f})")
    
def plot_logistic_probability_curve(feature='popularity_log', savepath='Classification/logistic_probability_curve.png'):
    """
    Logistic Regression probability curve for a single feature.
    - Target: hit_flag = 1 if revenue > median, else 0
    - Feature options (recommended): 'popularity_log', 'budget_log', 'vote_count_log'
    - Other numeric features are held at their median to isolate the curve.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    # Prepare data (similar to your classification prep)
    keep = (df['revenue'] > 0) & (df['budget'] > 0) & (df['vote_count'] >= 50)
    data = df.loc[keep, ['budget','revenue','popularity','vote_average','vote_count','runtime']].copy()

    # Log transforms
    data['budget_log'] = np.log1p(data['budget'])
    data['popularity_log'] = np.log1p(data['popularity'])
    data['vote_count_log'] = np.log1p(data['vote_count'])

    # Target: Hit vs Flop by median revenue
    rev_med = data['revenue'].median()
    data['hit_flag'] = (data['revenue'] > rev_med).astype(int)

    # Feature set (you can add/remove)
    feat_cols = ['budget_log','popularity_log','vote_average','vote_count_log','runtime']
    X = data[feat_cols].fillna(0)
    y = data['hit_flag']

    # Train/test split (train only needed to fit curve)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit logistic model
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)

    # Build a grid over the chosen feature while holding others at median
    medians = X_train.median()
    x_min, x_max = X_train[feature].quantile([0.01, 0.99])
    x_grid = np.linspace(x_min, x_max, 200)

    X_grid = pd.DataFrame({col: medians[col] for col in feat_cols}, index=range(len(x_grid)))
    X_grid[feature] = x_grid

    # Predict probabilities across the grid
    prob = log_reg.predict_proba(X_grid)[:, 1]

    # Scatter actual points (feature vs label) for context
    plt.figure(figsize=(8,5))
    plt.scatter(X_train[feature], y_train, s=15, alpha=0.35, label='Actual (train)', color='black')

    # Plot sigmoid probability curve
    plt.plot(x_grid, prob, linewidth=2.5, label='Predicted P( Hit )')

    # Labels & save
    xlabel = {
        'budget_log': 'log(1 + Budget)',
        'popularity_log': 'log(1 + Popularity)',
        'vote_count_log': 'log(1 + Vote Count)'
    }.get(feature, feature)

    plt.title(f'Probability of Hit vs {xlabel}')
    plt.xlabel(xlabel)
    plt.ylabel('Predicted Probability of Hit')
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath, dpi=120)
    plt.show()

    print(f"Saved probability curve to: {savepath}")

# ====================================================================
# MAIN ANALYSIS EXECUTION
# ====================================================================

print("\n" + "="*70)
print("MOVIE DATASET ANALYSIS - CLASSIFICATION")
print("="*70)


# Movie Hit Analysis
print("\n" + "-"*50)
print("Movie Hit ANALYSIS")
analyze_movie_hit_prediction()
print("-"*50)

# Movie Hit Analysis
print("\n" + "-"*50)
print("Movie Hit ANALYSIS")
# Example 1: curve vs popularity (cleanest sigmoid)
plot_logistic_probability_curve(feature='popularity_log')
print("-"*50)
# Example 2: curve vs budget
plot_logistic_probability_curve(feature='budget_log', savepath='Classification/logit_curve_budget.png')
print("-"*50)
# Example 3: curve vs vote count
plot_logistic_probability_curve(feature='vote_count_log', savepath='Classification/logit_curve_votes.png')

print("-"*50)
plot_roc_curve()

print("\n" + "="*70)
print("ANALYSIS COMPLETE - Check generated PNG files for visualizations")
print("="*70)
