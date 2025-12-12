import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import IPython.display


df = pd.read_excel("../Data/processed/firearm_data_cleaned.xlsx")
df.head()

# Basic Descriptive Statistics
# Categorical Variables Summary

df[["year", "rate", "deaths", "law_strength_score", "restrictive_laws", "permissive_laws"]].describe()

## Categorical Counts
print(df['year'].value_counts())
print(df['state_name'].value_counts())

# ## Distribution Visualizations

# histogram of death rates
fig, ax = plt.subplots(figsize = (10, 6))

ax.hist(df['rate'].dropna(), bins = 30, edgecolor = 'black', alpha = 0.7)
ax.axvline(df['rate'].mean(), color = 'r', linestyle = 'dashed', linewidth = 2,
           label = 'Mean')
ax.set_xlabel('Death Rate per 100,000', fontsize = 12)
ax.set_ylabel('Frequnecy', fontsize = 12)
ax.set_title('Distribution of Firearm Death Rates', fontsize = 14, fontweight = 'bold')
ax.legend()
plt.tight_layout()
plt.show()

# histogram of death counts
fig, ax = plt.subplots(figsize = (10, 6))

ax.hist(df['deaths'].dropna(), bins = 30, edgecolor = 'black', alpha = 0.7)
ax.axvline(df['deaths'].mean(), color = 'r', linestyle = 'dashed', linewidth = 2,
           label = 'Mean')
ax.set_xlabel('Number of Deaths', fontsize = 12)
ax.set_ylabel('Frequnecy', fontsize = 12)
ax.set_title('Distribution of Firearm Deaths', fontsize = 14, fontweight = 'bold')
ax.legend()
plt.tight_layout()
plt.show()

# histogram of law strength score
fig, ax = plt.subplots(figsize = (10, 6))

ax.hist(df['law_strength_score'].dropna(), bins = 30, edgecolor = 'black', alpha = 0.7)
ax.axvline(df['law_strength_score'].mean(), color = 'r', linestyle = 'dashed', linewidth = 2,
           label = 'Mean')
ax.set_xlabel('Law Strength Score', fontsize = 12)
ax.set_ylabel('Frequnecy', fontsize = 12)
ax.set_title('Distribution of Law Strength Scores', fontsize = 14, fontweight = 'bold')
ax.legend()
plt.tight_layout()
plt.show()

# histogram of restrictive laws
fig, ax = plt.subplots(figsize = (10, 6))

ax.hist(df['restrictive_laws'].dropna(), bins = 30, edgecolor = 'black', alpha = 0.7)
ax.axvline(df['restrictive_laws'].mean(), color = 'r', linestyle = 'dashed', linewidth = 2,
           label = 'Mean')
ax.set_xlabel('Number of Restrictive Laws', fontsize = 12)
ax.set_ylabel('Frequnecy', fontsize = 12)
ax.set_title('Distribution of Restrictive Laws', fontsize = 14, fontweight = 'bold')
ax.legend()
plt.tight_layout()
plt.show()

# histogram of permissive laws
fig, ax = plt.subplots(figsize = (10, 6))

ax.hist(df['permissive_laws'].dropna(), bins = 30, edgecolor = 'black', alpha = 0.7)
ax.axvline(df['permissive_laws'].mean(), color = 'r', linestyle = 'dashed', linewidth = 2,
           label = 'Mean')
ax.set_xlabel('Number of Permissive Laws', fontsize = 12)
ax.set_ylabel('Frequnecy', fontsize = 12)
ax.set_title('Distribution of Permissive Laws', fontsize = 14, fontweight = 'bold')
ax.legend()
plt.tight_layout()
plt.show()


# Boxplot of Year by Rate
sns.boxplot(x='year', y='rate', data=df)
plt.show()


# NA Values
df[['state_name','year', 'deaths', 'rate', 'law_strength_score', 'restrictive_laws', 'permissive_laws']].isna().sum()


# Correlation Heatmap
correlation_matrix = df[['year', 'rate', 'law_strength_score', 'restrictive_laws', 'permissive_laws']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

# Deaths Rates by Law Categories

# Create categories for analysis
df['restrictive_category'] = pd.cut(df['restrictive_laws'], 
                                               bins=3, 
                                               labels=['Low', 'Medium', 'High'])
df['permissive_category'] = pd.cut(df['permissive_laws'], 
                                              bins=3, 
                                              labels=['Low', 'Medium', 'High'])

# Boxplot: Death rates by restrictive law levels
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df, x='restrictive_category', y='rate', 
            palette='Greens', ax=ax, hue='restrictive_category')
ax.set_xlabel('Restrictive Laws Level', fontsize=12)
ax.set_ylabel('Death Rate per 100,000', fontsize=12)
ax.set_title('Death Rates by Restrictive Law Levels', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Boxplot: Death rates by permissive law levels
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df, x='permissive_category', y='rate', 
            palette='Reds', ax=ax, hue='permissive_category')
ax.set_xlabel('Permissive Laws Level', fontsize=12)
ax.set_ylabel('Death Rate per 100,000', fontsize=12)
ax.set_title('Death Rates by Permissive Law Levels', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Side-by-side violin plots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.violinplot(data=df, x='restrictive_category', y='rate', 
               palette='Greens', ax=axes[0], hue = 'restrictive_category')
axes[0].set_xlabel('Restrictive Laws Level', fontsize=12)
axes[0].set_ylabel('Death Rate per 100,000', fontsize=12)
axes[0].set_title('Death Rates by Restrictive Laws', fontsize=13, fontweight='bold')

sns.violinplot(data=df, x='permissive_category', y='rate', 
               palette='Reds', ax=axes[1], hue = 'permissive_category')
axes[1].set_xlabel('Permissive Laws Level', fontsize=12)
axes[1].set_ylabel('Death Rate per 100,000', fontsize=12)
axes[1].set_title('Death Rates by Permissive Laws', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.show()

# Temporal Trends
# Overall trend over time
temporal_summary = df.groupby('year').agg({
    'rate': 'mean',
    'restrictive_laws': 'mean',
    'permissive_laws': 'mean',
    'law_strength_score': 'mean'
}).reset_index()

# Death rate over time
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(temporal_summary['year'], temporal_summary['rate'], 
        color='darkred', linewidth=2, marker='o', markersize=8, label='Death Rate')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Average Death Rate per 100,000', fontsize=12)
ax.set_title('Average Firearm Death Rate Over Time (2014-2023)', 
             fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Law trends over time
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(temporal_summary['year'], temporal_summary['restrictive_laws'], 
        color='darkgreen', linewidth=2, marker='o', markersize=8, label='Restrictive Laws')
ax.plot(temporal_summary['year'], temporal_summary['permissive_laws'], 
        color='coral', linewidth=2, marker='s', markersize=8, label='Permissive Laws')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Average Number of Laws', fontsize=12)
ax.set_title('Restrictive vs Permissive Laws Over Time', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Combined view: All three metrics
fig, ax1 = plt.subplots(figsize=(14, 7))

# Death rate on primary y-axis
color1 = 'darkred'
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Death Rate per 100,000', color=color1, fontsize=12)
line1 = ax1.plot(temporal_summary['year'], temporal_summary['rate'], 
                 color=color1, linewidth=2.5, marker='o', markersize=8, 
                 label='Death Rate')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, alpha=0.3)

# Law counts on secondary y-axis
ax2 = ax1.twinx()
color2 = 'darkgreen'
color3 = 'coral'
ax2.set_ylabel('Number of Laws', fontsize=12)
line2 = ax2.plot(temporal_summary['year'], temporal_summary['restrictive_laws'], 
                 color=color2, linewidth=2, marker='s', markersize=8, 
                 label='Restrictive Laws', linestyle='--')
line3 = ax2.plot(temporal_summary['year'], temporal_summary['permissive_laws'], 
                 color=color3, linewidth=2, marker='^', markersize=8, 
                 label='Permissive Laws', linestyle='--')

# Combine legends
lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', fontsize=11)

plt.title('Death Rates and Gun Laws Over Time', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

# Gun Law Analytics
# Restrictive vs Permissive Laws scatter with death rate coloring
fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(df['restrictive_laws'], 
                     df['permissive_laws'],
                     c=df['rate'], s=df['rate']*3,
                     cmap='RdYlGn_r', alpha=0.6, edgecolors='black', linewidth=0.5)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Death Rate per 100,000', fontsize=12)
ax.set_xlabel('Number of Restrictive Laws', fontsize=12)
ax.set_ylabel('Number of Permissive Laws', fontsize=12)
ax.set_title('Restrictive vs Permissive Laws by Death Rate', 
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Separate scatter plots for each law type vs death rate
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Restrictive laws vs death rate
axes[0].scatter(df['restrictive_laws'], df['rate'], 
                alpha=0.5, color='darkgreen', s=50, edgecolors='black', linewidth=0.5)

# Add regression line
from scipy.stats import linregress
mask1 = ~(df['restrictive_laws'].isna() | df['rate'].isna())
slope1, intercept1, r_value1, p_value1, std_err1 = linregress(
    df.loc[mask1, 'restrictive_laws'], 
    df.loc[mask1, 'rate']
)
line_x1 = np.array([df['restrictive_laws'].min(), 
                    df['restrictive_laws'].max()])
line_y1 = slope1 * line_x1 + intercept1
axes[0].plot(line_x1, line_y1, 'b-', linewidth=2, 
             label=f'Linear fit (R²={r_value1**2:.3f})')
axes[0].set_xlabel('Number of Restrictive Laws', fontsize=12)
axes[0].set_ylabel('Death Rate per 100,000', fontsize=12)
axes[0].set_title('Restrictive Laws vs Death Rate', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Permissive laws vs death rate
axes[1].scatter(df['permissive_laws'], df['rate'], 
                alpha=0.5, color='coral', s=50, edgecolors='black', linewidth=0.5)

# Add regression line
mask2 = ~(df['permissive_laws'].isna() | df['rate'].isna())
slope2, intercept2, r_value2, p_value2, std_err2 = linregress(
    df.loc[mask2, 'permissive_laws'], 
    df.loc[mask2, 'rate']
)
line_x2 = np.array([df['permissive_laws'].min(), 
                    df['permissive_laws'].max()])
line_y2 = slope2 * line_x2 + intercept2
axes[1].plot(line_x2, line_y2, 'b-', linewidth=2, 
             label=f'Linear fit (R²={r_value2**2:.3f})')
axes[1].set_xlabel('Number of Permissive Laws', fontsize=12)
axes[1].set_ylabel('Death Rate per 100,000', fontsize=12)
axes[1].set_title('Permissive Laws vs Death Rate', fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print correlation coefficients
print(f"\nCorrelation between Restrictive Laws and Death Rate: {df['restrictive_laws']\
                                                                .corr(df['rate']):.4f}")
print(f"Correlation between Permissive Laws and Death Rate: {df['permissive_laws']\
                                                             .corr(df['rate']):.4f}")


# Law Strength Analysis
# Law Strength vs Death Rate
fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(df['law_strength_score'], df['rate'], 
                     c=df['restrictive_laws'], s=100,
                     cmap='Greens', alpha=0.6, edgecolors='black', linewidth=0.5)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Number of Restrictive Laws', fontsize=12)

# Add regression line
mask = ~(df['law_strength_score'].isna() | df['rate'].isna())
slope, intercept, r_value, p_value, std_err = linregress(
    df.loc[mask, 'law_strength_score'], 
    df.loc[mask, 'rate']
)
line_x = np.array([df['law_strength_score'].min(), 
                   df['law_strength_score'].max()])
line_y = slope * line_x + intercept
ax.plot(line_x, line_y, 'b-', linewidth=2, label=f'Linear fit (R²={r_value**2:.3f})')

ax.set_xlabel('Law Strength Score', fontsize=12)
ax.set_ylabel('Death Rate per 100,000', fontsize=12)
ax.set_title('Gun Law Strength vs Firearm Death Rate\n(colored by Restrictive Laws)', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# Density plot comparison
fig, ax = plt.subplots(figsize=(12, 6))

# Create quartiles for restrictive laws
df['restrictive_quartile'] = pd.qcut(df['restrictive_laws'], 
                                                 q=4, 
                                                 labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])

for quartile in df['restrictive_quartile'].unique():
    if pd.notna(quartile):
        data = df[df['restrictive_quartile'] == quartile]['rate']
        data.dropna().plot(kind='density', ax=ax, label=quartile, linewidth=2, alpha=0.7)

ax.set_xlabel('Death Rate per 100,000', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Distribution of Death Rates by Restrictive Law Quartiles', 
             fontsize=14, fontweight='bold')
ax.legend(title='Restrictive Laws', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Specific Law Categories Analysis
# Get strength variables
strength_cols = [col for col in df.columns if col.startswith('strength_')]

# Calculate average strength by category
strength_avg = df[strength_cols].mean().sort_values(ascending=False)
strength_avg.index = strength_avg.index.str.replace('strength_', '').str.replace('_', ' ').str.title()

# Plot
fig, ax = plt.subplots(figsize=(10, 8))
strength_avg.plot(kind='barh', ax=ax, color='steelblue')
ax.set_xlabel('Average Strength Score', fontsize=12)
ax.set_ylabel('Law Category', fontsize=12)
ax.set_title('Average Strength Score by Law Category', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# State-Level Comparisons
# Get most recent year data
recent_year = df['year'].max()
recent_data = df[df['year'] == recent_year]

# Top 10 states by death rate
top_10 = recent_data.nlargest(10, 'rate')
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(range(len(top_10)), top_10['rate'].values, color='coral')
ax.set_yticks(range(len(top_10)))
ax.set_yticklabels(top_10['state_name'].values)
ax.set_xlabel('Death Rate per 100,000', fontsize=12)
ax.set_title(f'Top 10 States by Firearm Death Rate ({int(recent_year)})', 
             fontsize=14, fontweight='bold')
ax.invert_yaxis()
plt.tight_layout()
plt.show()

# Bottom 10 states
bottom_10 = recent_data.nsmallest(10, 'rate')
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(range(len(bottom_10)), bottom_10['rate'].values, color='lightgreen')
ax.set_yticks(range(len(bottom_10)))
ax.set_yticklabels(bottom_10['state_name'].values)
ax.set_xlabel('Death Rate per 100,000', fontsize=12)
ax.set_title(f'Bottom 10 States by Firearm Death Rate ({int(recent_year)})', 
             fontsize=14, fontweight='bold')
ax.invert_yaxis()
plt.tight_layout()
plt.show()

# States with most permissive laws
top_permissive = recent_data.nlargest(10, 'permissive_laws')
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(range(len(top_permissive)), top_permissive['permissive_laws'].values, color='coral')
ax.set_yticks(range(len(top_permissive)))
ax.set_yticklabels(top_permissive['state_name'].values)
ax.set_xlabel('Number of Permissive Laws', fontsize=12)
ax.set_title(f'Top 10 States by Permissive Laws ({int(recent_year)})', 
             fontsize=14, fontweight='bold')
ax.invert_yaxis()
plt.tight_layout()
plt.show()

# States with most restrictive laws
top_restrictive = recent_data.nlargest(10, 'restrictive_laws')
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(range(len(top_restrictive)), top_restrictive['restrictive_laws'].values, color='darkgreen')
ax.set_yticks(range(len(top_restrictive)))
ax.set_yticklabels(top_restrictive['state_name'].values)
ax.set_xlabel('Number of Restrictive Laws', fontsize=12)
ax.set_title(f'Top 10 States by Restrictive Laws ({int(recent_year)})', 
             fontsize=14, fontweight='bold')
ax.invert_yaxis()
plt.tight_layout()
plt.show()

# ## Correlation Analysis
# Select correlation variables
cor_vars = ['rate', 'deaths', 'law_strength_score', 'restrictive_laws', 
            'permissive_laws', 'restrictive_ratio', 'permissive_ratio'] + strength_cols
cor_data = df[cor_vars].dropna(axis=1, how='all')

# Calculate correlation matrix
cor_matrix = cor_data.corr()

# Visualize correlation matrix
fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(cor_matrix, dtype=bool))
sns.heatmap(cor_matrix, mask=mask, annot=False, cmap='coolwarm', 
            center=0, square=True, linewidths=0.5, 
            cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Correlation Matrix: Death Rates and Gun Laws', 
             fontsize=14, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()
plt.show()

# Top correlations with rate
rate_cors = cor_matrix['rate'].drop('rate').abs().sort_values(ascending=False).head(15)
print("\nTop 15 correlations with death rate:")
correlation_df = pd.DataFrame({
    'Variable': rate_cors.index,
    'Correlation': cor_matrix['rate'][rate_cors.index].values
}).round(4)
print(correlation_df)

# Top correlations with rate
rate_cors = cor_matrix['rate'].drop('rate').abs().sort_values(ascending=False).head(15)
print("\nTop 15 correlations with death rate:")
correlation_df = pd.DataFrame({
    'Variable': rate_cors.index,
    'Correlation': cor_matrix['rate'][rate_cors.index].values
}).round(4)
print(correlation_df)

# Highlight permissive and restrictive laws
print("\n" + "="*50)
print(f"Restrictive Laws correlation with death rate: {cor_matrix.loc['restrictive_laws', 'rate']:.4f}")
print(f"Permissive Laws correlation with death rate: {cor_matrix.loc['permissive_laws', 'rate']:.4f}")
print(f"Law Strength Score correlation with death rate: {cor_matrix.loc['law_strength_score', 'rate']:.4f}")
print("="*50)

# Year-Over-Year Changes
# Rate changes
if 'rate_change' in df.columns:
    fig, ax = plt.subplots(figsize=(10, 6))
    rate_changes = df['rate_change'].dropna()
    ax.hist(rate_changes, bins=30, color='purple', alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No Change')
    ax.set_xlabel('Change in Death Rate', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Year-over-Year Rate Changes', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.show()

# Law strength changes
if 'law_strength_change' in df.columns:
    fig, ax = plt.subplots(figsize=(10, 6))
    law_changes = df['law_strength_change'].dropna()
    ax.hist(law_changes, bins=30, color='purple', alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No Change')
    ax.set_xlabel('Change in Law Strength Score', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Year-over-Year Law Strength Changes', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.show()

# ## Summary Statistics Table
# Overall summary
print("\nOverall Summary Statistics:")
print(df[['rate', 'deaths', 'restrictive_laws', 
                    'permissive_laws', 'law_strength_score']].describe())

# By restrictive law quartiles
summary_restrictive = df.groupby('restrictive_quartile').agg({
    'rate': ['count', 'mean', 'std', 'min', 'max'],
    'deaths': 'mean',
    'permissive_laws': 'mean',
    'law_strength_score': 'mean'
}).round(2)

print("\nSummary Statistics by Restrictive Law Quartiles:")
print(summary_restrictive)

# By permissive law quartiles
df['permissive_quartile'] = pd.qcut(df['permissive_laws'], 
                                                q=4, 
                                                labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])

summary_permissive = df.groupby('permissive_quartile').agg({
    'rate': ['count', 'mean', 'std', 'min', 'max'],
    'deaths': 'mean',
    'restrictive_laws': 'mean',
    'law_strength_score': 'mean'
}).round(2)

print("\nSummary Statistics by Permissive Law Quartiles:")
print(summary_permissive)

strength_col = [col for col in df.columns if col.startswith('strength_')]

recent_year = df['year'].max()
recent_data = df[df['year'] == recent_year]

correlations = recent_data[strength_col + ['rate']].corr()['rate'].drop('rate')

cor_df = pd.DataFrame({
    'law_type': correlations.index,
    'correlation': correlations.values
})

cor_df['law_type_clean'] = (cor_df['law_type']
                            .str.replace('strength_', '')
                            .str.replace('_', ' ')
                            .str.title())

cor_df = cor_df.sort_values('correlation')

print("\nCorrelations with Death Rate for Most Recent Year:")
print(cor_df)   

# Top protective and harmful laws 

top_protective = cor_df.head(10)
top_harmful = cor_df.tail(10)

print("\nTop 10 Most Protective Laws:")
print(top_protective)

print("\nTop 10 Most Harmful Laws:")
print(top_harmful)

# Visualizations of Top Protective Laws

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(top_protective['law_type_clean'], top_protective['correlation'])
ax.set_xlabel('Correlation with Death Rate', fontsize=12)
ax.set_ylabel('Law Type', fontsize=12)
ax.set_title('Top 10 Most Protective Laws', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


