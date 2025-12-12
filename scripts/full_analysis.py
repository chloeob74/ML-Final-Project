# Core Data Manipulation
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objs as go

# Preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Regression Models
from sklearn.linear_model import LinearRegression, RidgeCV, Lasso
from sklearn.neighbors import KNeighborsRegressor

# Classification Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Clustering
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# Evaluation Metrics
from sklearn.metrics import (
    mean_squared_error, r2_score, 
    accuracy_score, classification_report, confusion_matrix
)

# Statistical Tests
from scipy import stats
from scipy.stats import f_oneway, pearsonr

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

print("All libraries imported successfully.")

# Load the merged dataset
df = pd.read_csv("../Data/processed/firearm_data_cleaned.csv")


print(f"Dataset Shape: {df.shape[0]} observations x {df.shape[1]} variables")
print(f"Time Period: {df['year'].min()} to {df['year'].max()}")
print(f"Number of States: {df['state'].nunique()}")
print(f"\nTarget Variable: 'rate' (firearm deaths per 100,000 population)")
print(f"  Mean: {df['rate'].mean():.2f}")
print(f"  Std:  {df['rate'].std():.2f}")
print(f"  Range: {df['rate'].min():.1f} – {df['rate'].max():.1f}")


# Some analysis has problems with missing years, so exclude DC which has only 2022, 2023
noDC = df[df['state'] != 'District of Columbia']

# Identify law strength features (our primary predictors)
law_features = [col for col in df.columns if col.startswith('strength_')]
permissive_classes = [col for col in df.columns if col.startswith('class_permissive_')]
restrictive_classes = [col for col in df.columns if col.startswith('class_restrictive_')]
feature_classes = permissive_classes + restrictive_classes

print(f"Law Strength Features ({len(law_features)} categories):")
for i, feat in enumerate(law_features, 1):
    clean_name = feat.replace('strength_', '').replace('_', ' ').title()
    print(f"  {i:2}. {clean_name}")


# kitchen sink models

data1 = df[['year', 'state', 'rate', 'deaths', 'law_strength_score',
                    'restrictive_laws', 'permissive_laws', 'total_law_changes',                             
                    'law_strength_change', 'unique_law_classes']].dropna()
    # state is enough, omitting state_name
    # omitting rate_change
data1 = data1.query("state!='District of Columbia'")
    # there is only one datapoint for District of Columbia, flummoxes train_test_split. dropping.


data2 = df[['rate', 'state', 'strength_background_checks', 'strength_carrying_a_concealed_weapon_ccw', 
                    'strength_castle_doctrine', 'strength_dealer_license', 'strength_firearm_sales_restrictions',
                    'strength_local_laws_preempted_by_state', 'strength_minimum_age','strength_prohibited_possessor',            
                    'strength_registration', 'strength_waiting_period', 'strength_firearm_removal_at_scene_of_domestic_violence',
                    'strength_firearms_in_college_university', 'strength_child_access_laws', 'strength_gun_trafficking', 
                    'strength_open_carry', 'strength_required_reporting_of_lost_or_stolen_firearms',
                    'strength_safety_training_required', 'strength_untraceable_firearms', 'strength_permit_to_purchase',
                    'strength_firearms_in_k_12_educational_settings']].dropna()
data2 = data2.query("state!='District of Columbia'")

bigX_1 = data1[['year', 'state', 'deaths', 
                    'restrictive_laws', 'permissive_laws',                             
                    'law_strength_change', 'unique_law_classes']]
    # state is enough, omitting state_name
    # omitting rate_change

bigX_2 = data2[['state', 'strength_background_checks', 'strength_carrying_a_concealed_weapon_ccw', 
                    'strength_castle_doctrine', 'strength_dealer_license', 'strength_firearm_sales_restrictions',
                    'strength_local_laws_preempted_by_state', 'strength_minimum_age','strength_prohibited_possessor',            
                    'strength_registration', 'strength_waiting_period', 'strength_firearm_removal_at_scene_of_domestic_violence',
                    'strength_firearms_in_college_university', 'strength_child_access_laws', 'strength_gun_trafficking', 
                    'strength_open_carry', 'strength_required_reporting_of_lost_or_stolen_firearms',
                    'strength_safety_training_required', 'strength_untraceable_firearms', 'strength_permit_to_purchase',
                    'strength_firearms_in_k_12_educational_settings']]
bigY1 = data1['rate']
bigY2 = data2['rate']

categories1 = ['year', 'state']
numers1 = ['deaths', 'restrictive_laws', 'permissive_laws',                           
                    'law_strength_change', 'unique_law_classes']

categories2 = ['state']
numers2 = ['strength_background_checks', 'strength_carrying_a_concealed_weapon_ccw', 
                    'strength_castle_doctrine', 'strength_dealer_license', 'strength_firearm_sales_restrictions',
                    'strength_local_laws_preempted_by_state', 'strength_minimum_age','strength_prohibited_possessor',            
                    'strength_registration', 'strength_waiting_period', 'strength_firearm_removal_at_scene_of_domestic_violence',
                    'strength_firearms_in_college_university', 'strength_child_access_laws', 'strength_gun_trafficking', 
                    'strength_open_carry', 'strength_required_reporting_of_lost_or_stolen_firearms',
                    'strength_safety_training_required', 'strength_untraceable_firearms', 'strength_permit_to_purchase',
                    'strength_firearms_in_k_12_educational_settings']


x_train1, x_test1, y_train1, y_test1 = train_test_split(bigX_1, bigY1, test_size = 0.2, random_state = 123)
x_train2, x_test2, y_train2, y_test2 = train_test_split(bigX_2, bigY2, test_size = 0.2, random_state = 123)


xform1 = ColumnTransformer(transformers = [("encoder1", OneHotEncoder(drop='first'), categories1),
                                           ("numeric1", "passthrough", numers1)])
xform2 = ColumnTransformer(transformers = [("encoder2", OneHotEncoder(drop='first'), categories2),
                                           ("numeric2", "passthrough", numers2)])

sinkMod1 = Pipeline(steps = [("transformer1", xform1), ("model1", LinearRegression())])
sinkMod2 = Pipeline(steps = [("transformer2", xform2), ("model2", LinearRegression())])

sinkMod1.fit(x_train1, y_train1)
sinkMod2.fit(x_train2, y_train2)


predictions1 = sinkMod1.predict(x_test1)
mse_calc1 = mean_squared_error(y_test1, predictions1)
rmse1 = mse_calc1 ** 0.5
rmse1


predictions2 = sinkMod2.predict(x_test2)
mse_calc2 = mean_squared_error(y_test2, predictions2)
rmse2 = mse_calc2 ** 0.5
rmse2

"""
the following code (calculating adjusted r^2) was produced by generative AI (Claude by Anthropic)) 
in response to a direct request of how to calculate adjusted r^2 for a multi-linear model built with
scikitlearn's Pipeline.
"""

r2_1 = sinkMod1.score(x_test1, y_test1)
n_1 = x_test1.shape[0]
p_1 = x_test1.shape[1]
adj_r2_1 = 1 - (1 - r2_1) * (n_1-1) / (n_1 - p_1 - 1)
adj_r2_1


r2_2 = sinkMod2.score(x_test2, y_test2)
n_2 = x_test2.shape[0]
p_2 = x_test2.shape[1]
adj_r2_2 = 1 - (1 - r2_2) * (n_2 - 1) / (n_2 - p_2 -1)
adj_r2_2


"""
the following code (capturing coefficients from the model and putting them into a dataframe, the latter several cells down) was 
produced by generative AI (Claude by Anthropic)) in response to a direct request of how to determine coefficients for 
a multi-linear model built with scikitlearn's OneHotEncoder.
"""

first_set = xform1.get_feature_names_out()
second_set = xform2.get_feature_names_out()

firstMod = sinkMod1.named_steps['model1']
#print(f"Intercept for first model: {firstMod.intercept_}")
#rint(f"Coefficients for first model: {firstMod.coef_}")
secondMod = sinkMod2.named_steps['model2']
#print(f"Intercept for second model: {secondMod.intercept_}")
#print(f"Coefficients for second model: {secondMod.coef_}")

coef_df_1 = pd.DataFrame({'feature': first_set, 'coefficient': firstMod.coef_})
#coef_df_1
coef_df_2 = pd.DataFrame({'feature': second_set, 'coefficient': secondMod.coef_})
#coef_df_2

plotData1 = pd.DataFrame({"actual": y_test1, "predicted": predictions1})
plotData2 = pd.DataFrame({"actual": y_test2, "predicted": predictions2})

plotData1.to_csv('basic_output_data1.csv')
plotData2.to_csv('basic_output_data2.csv')

px.scatter(plotData1, x="predicted", y="actual", trendline='ols')
px.scatter(plotData2, x="predicted", y="actual", trendline='ols')

# Ridge Regression

"""
Apply Ridge regression (IAW notebook 12 and guidance from genAI)

1.  Make a set of lambdas
2.  Re-preprocess data >> StandardScaler for numerical.
3.  Use RidgeCV object to train model (via Pipeline)
4.  Test model
5.  Evaluate performance

NOTE: doing Ridge but NOT Lasso because Lasso is strong when only a few predictors matter;
This project attempts to model a complex phenomenon, so this possiblity is dismissed.

"""

# Ridge regression with multiple alpha (lambda) values
lambdas = np.logspace(-2, 6, 100)

# re-preprocess data
xform1_1 = ColumnTransformer([('nums', StandardScaler(), numers1),
                              ('cats', OneHotEncoder(drop='first'), categories1)])
xform2_2 = ColumnTransformer([("nums2", StandardScaler(), numers2),
                              ("cats2", OneHotEncoder(drop="first"), categories2)])

# Pipeline >> train model
updated_model1 = Pipeline(steps=[('transformer1_1', xform1_1), 
                                ("model1_1", RidgeCV(alphas=lambdas, cv=10, scoring='neg_mean_squared_error'))])
updated_model2 = Pipeline(steps=[("transformer2_2", xform2_2), 
                                 ("model2_2", RidgeCV(alphas=lambdas, cv=10, scoring='neg_mean_squared_error'))])

updated_model1.fit(x_train1, y_train1)
updated_model2.fit(x_train2, y_train2)

# Test Model
predictions1_1 = updated_model1.predict(x_test1)
predictions2_2 = updated_model2.predict(x_test2)

# Evaluate performance

plotData1_1 = pd.DataFrame({"actual": y_test1, "predicted": predictions1_1})
plotData2_2 = pd.DataFrame({"actual": y_test2, "predicted": predictions2_2})

px.scatter(plotData1_1, x='predicted', y='actual', trendline='ols')

plotData1_1.to_csv('ridge_output_data1.csv')
plotData2_2.to_csv('ridge_output_data2.csv')

px.scatter(plotData2_2, x='predicted', y='actual', trendline='ols')

# and the metrics

# inputs to .score are both x_test and y_test
r2_1_1 = updated_model1.score(x_test1, y_test1)
n = x_test2.shape[0]
p = x_test2.shape[1]
adj_r2_1_1 = 1 - (1 - r2_1_1) * (n - 1) / (n - p -1)
adj_r2_1_1

r2_2_2 = updated_model2.score(x_test2, y_test2)
n = x_test2.shape[0]
p = x_test2.shape[1]
adj_r2_2_2 = 1 - (1 - r2_2_2) * (n - 1) / (n - p -1)
adj_r2_2_2

mse_calc1_1 = mean_squared_error(y_test1, predictions1_1)
root_mse1_1 = mse_calc1_1 ** 0.5
root_mse1_1

mse_calc2_2 = mean_squared_error(y_test2, predictions2_2)
root_mse2_2 = mse_calc2_2 ** 0.5
root_mse2_2

# Principal Component Analysis 

data1 = df[['year', 'state', 'rate', 'deaths',
                    'restrictive_laws', 'permissive_laws',                             
                    'law_strength_change', 'unique_law_classes']].dropna()
    # state is enough, omitting state_name
    # omitting rate_change
data1 = data1.query("state!='District of Columbia'")
    # there is only one datapoint for District of Columbia, flummoxes train_test_split. dropping.

bigX_1 = data1[['year', 'state', 'deaths',
                    'restrictive_laws', 'permissive_laws',                             
                    'law_strength_change', 'unique_law_classes']]
    # state is enough, omitting state_name
    # omitting rate_change

bigY1 = data1['rate']

categories1 = ['year', 'state']
numers1 = ['deaths', 'restrictive_laws', 'permissive_laws',                           
                    'law_strength_change', 'unique_law_classes']

# dummify 'state'
#X = pd.get_dummies(data2["smoker"], drop_first = True, dtype = float)
bigX_11 = pd.get_dummies(bigX_1['state'], drop_first=True, dtype=float)
bigX_12 = pd.concat([bigX_1, bigX_11], axis=1)
bigX_12['AK'] = bigX_12['state'] == 'AK'
bigX_1 = bigX_12.drop('state', axis=1)
# X.columns = X.columns.astype(str)

# dummify 'year'
temp = pd.get_dummies(bigX_1['year'], drop_first=True, dtype=float)
temp2 = pd.concat([bigX_1, temp], axis=1)
temp2[2015] = temp2['year']==2015
bigX_1 = temp2.drop('year', axis=1)

bigX_1.columns = bigX_1.columns.astype(str)

data2 = df[['rate', 'year', 'state', 'strength_background_checks', 'strength_carrying_a_concealed_weapon_ccw', 
                    'strength_castle_doctrine', 'strength_dealer_license', 'strength_firearm_sales_restrictions',
                    'strength_local_laws_preempted_by_state', 'strength_minimum_age','strength_prohibited_possessor',            
                    'strength_registration', 'strength_waiting_period', 'strength_firearm_removal_at_scene_of_domestic_violence',
                    'strength_firearms_in_college_university', 'strength_child_access_laws', 'strength_gun_trafficking', 
                    'strength_open_carry', 'strength_required_reporting_of_lost_or_stolen_firearms',
                    'strength_safety_training_required', 'strength_untraceable_firearms', 'strength_permit_to_purchase',
                    'strength_firearms_in_k_12_educational_settings']].dropna()
data2 = data2.query("state!='District of Columbia'")

bigX_2 = data2[['state', 'year', 'strength_background_checks', 'strength_carrying_a_concealed_weapon_ccw', 
                    'strength_castle_doctrine', 'strength_dealer_license', 'strength_firearm_sales_restrictions',
                    'strength_local_laws_preempted_by_state', 'strength_minimum_age','strength_prohibited_possessor',            
                    'strength_registration', 'strength_waiting_period', 'strength_firearm_removal_at_scene_of_domestic_violence',
                    'strength_firearms_in_college_university', 'strength_child_access_laws', 'strength_gun_trafficking', 
                    'strength_open_carry', 'strength_required_reporting_of_lost_or_stolen_firearms',
                    'strength_safety_training_required', 'strength_untraceable_firearms', 'strength_permit_to_purchase',
                    'strength_firearms_in_k_12_educational_settings']]

categories2 = ['state', 'year']
numers2 = ['strength_background_checks', 'strength_carrying_a_concealed_weapon_ccw', 
                    'strength_castle_doctrine', 'strength_dealer_license', 'strength_firearm_sales_restrictions',
                    'strength_local_laws_preempted_by_state', 'strength_minimum_age','strength_prohibited_possessor',            
                    'strength_registration', 'strength_waiting_period', 'strength_firearm_removal_at_scene_of_domestic_violence',
                    'strength_firearms_in_college_university', 'strength_child_access_laws', 'strength_gun_trafficking', 
                    'strength_open_carry', 'strength_required_reporting_of_lost_or_stolen_firearms',
                    'strength_safety_training_required', 'strength_untraceable_firearms', 'strength_permit_to_purchase',
                    'strength_firearms_in_k_12_educational_settings']

bigY2 = data2['rate']

# dummify 'state'
temp = pd.get_dummies(bigX_2['state'], drop_first=True, dtype=float)
temp2 = pd.concat([bigX_2, temp], axis=1)
temp2['AK'] = temp2['state'] == 'AK'
bigX_2 = temp2.drop('state', axis=1)

# dummify 'year'
temp = pd.get_dummies(bigX_2['year'], drop_first=True, dtype=float)
temp2 = pd.concat([bigX_2, temp], axis=1)
temp2[2015] = temp2['year']==2015
bigX_2 = temp2.drop('year', axis=1)

bigX_2.columns = bigX_2.columns.astype(str)

x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(bigX_1, bigY1, test_size=0.2, random_state=123)
x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(bigX_2, bigY2, test_size=0.2, random_state=123)

"""
Apply PCA (IAW notebook 17 and guidance from genAI)
"""
# center TRAINING data and calculate training data mean
x_train_1 = x_train_1.astype(float)
x_train_2 = x_train_2.astype(float)

mu_train1 = x_train_1.mean()
mu_train2 = x_train_2.mean()

Xc_1t = x_train_1 - mu_train1
Xc_2t = x_train_2 - mu_train2

# generate variance-covariance matrix
cov_1 = np.cov(Xc_1t.T)
cov_2 = np.cov(Xc_2t.T)

# eigendecomposition >> eigenvalues/eigenvectors
eigvals_1, eigvecs_1 = np.linalg.eig(cov_1)
eigvals_2, eigvecs_2 = np.linalg.eig(cov_2)

# sort the eigenvalues and eigenvectors
order1 = np.argsort(eigvals_1)[::-1]
order2 = np.argsort(eigvals_2)[::-1]

eigval_1_sorted = eigvals_1[order1]
eigval_2_sorted = eigvals_2[order2]

eigvec_1_sorted = eigvecs_1[order1]
eigvec_2_sorted = eigvecs_2[order2]

# isolate desired eigenvectors >> each column is one eigenvector
candidates1 = eigvec_1_sorted[:,:4]
candidates2 = eigvec_2_sorted[:,:4]

# project (centered) training data
pc_train_1 = Xc_1t @ candidates1
pc_train_2 = Xc_2t @ candidates2

# center test data with training data mean
Xc_1test = x_test_1 - mu_train1
Xc_2test = x_test_2 - mu_train2

# project (ceentered) test data
pc_test_1 = Xc_1test @ candidates1
pc_test_2 = Xc_2test @ candidates2

# regress (train model) >> LinearRegression() >> model.fit()
pc_model1 = LinearRegression()
pc_model1.fit(pc_train_1, y_train_1)

pc_model2 = LinearRegression()
pc_model2.fit(pc_train_2, y_train_2)

# predict (test model) using centered test data >> model.predict()
predictions1 = pc_model1.predict(pc_test_1)
predictions2 = pc_model2.predict(pc_test_2)

# evaluate performance
pc_data1 = pd.DataFrame({"actual": y_test_1, "predicted": predictions1})
pc_data2 = pd.DataFrame({"actual": y_test_2, "predicted": predictions2})

px.scatter(pc_data1, x='predicted', y='actual', trendline='ols')

px.scatter(pc_data2, x='predicted', y='actual', trendline='ols')

pc_data1.to_csv('../dashboard/pca_output_data1.csv')
pc_data2.to_csv('../dashboard/pca_output_data2.csv'
                
# inputs to .score are both x_test and y_test, in this case the centered X test data
r2_1 = pc_model1.score(pc_test_1, y_test_1)
n = x_test_1.shape[0]
p = x_test_1.shape[1]
adj_r2_1 = 1 - (1 - r2_1) * (n - 1) / (n - p -1)
print(f"Adjusted r^2 for PC model 1: {adj_r2_1}")

r2_2 = pc_model2.score(pc_test_2, y_test_2)
n = x_test_2.shape[0]
p = x_test_2.shape[1]
adj_r2_2 = 1 - (1 - r2_2) * (n - 1) / (n - p -1)
print(f"Adjusted r^2 for PC model 2: {adj_r2_2}")

mse_calc1_1 = mean_squared_error(y_test_1, predictions1)
rmse1_1 = mse_calc1_1 ** 0.5
rmse1_1

mse_calc2_2 = mean_squared_error(y_test_2, predictions2)
rmse2_2 = mse_calc2_2 ** 0.5
rmse2_2


# Question 2
# Identify which laws are restrictive (positive) vs permissive (negative)
recent_data = df[df['year'] == df['year'].max()][['state_name', 'rate'] + law_features].dropna()

law_type_info = []
for col in law_features:
    mean_val = recent_data[col].mean()
    law_type = 'Restrictive' if mean_val >= 0 else 'Permissive'
    clean_name = col.replace('strength_', '').replace('_', ' ').title()
    law_type_info.append({'Law': clean_name, 'Type': law_type, 'Mean Value': mean_val})

law_type_df = pd.DataFrame(law_type_info).sort_values('Mean Value', ascending=False)

print("Law Categories by Type:")
print("="*60)
print(f"\nRESTRICTIVE LAWS (positive values, add to strength score):")
restrictive = law_type_df[law_type_df['Type'] == 'Restrictive']
for _, row in restrictive.iterrows():
    print(f"  • {row['Law']:45} (avg: {row['Mean Value']:+.2f})")

print(f"\nPERMISSIVE LAWS (negative values, subtract from strength score):")
permissive = law_type_df[law_type_df['Type'] == 'Permissive']
for _, row in permissive.iterrows():
    print(f"  • {row['Law']:45} (avg: {row['Mean Value']:+.2f})")

print(f"\nTotal: {len(restrictive)} restrictive, {len(permissive)} permissive law categories")

# Calculate law correlations with death rate
correlations = recent_data[law_features + ['rate']].corr()['rate'].drop('rate')
cor_df = pd.DataFrame({
    'Law Type': [f.replace('strength_', '').replace('_', ' ').title() for f in correlations.index],
    'Correlation': correlations.values
}).sort_values('Correlation', key=abs, ascending=False)


# Calculate feature correlations with death rate
# Exclude classes with zero variance since they have no predictive power, and therefore cause nans in the correlation matrix
non_zero_v_feature_classes = df[feature_classes].std()[df[feature_classes].std() != 0].index.to_list()

correlations = noDC[non_zero_v_feature_classes + ['rate']].corr()['rate'].drop('rate')
cor2_df = pd.DataFrame({
    'Law Type': [f.replace('class_', '').replace('_', ' ').title() for f in correlations.index],
    'Correlation': correlations.values
}).sort_values(by='Correlation', key=abs, ascending=False)


# Visualization

cor_df['pos'] = cor_df['Correlation'] > 0
fig1a = px.bar(cor_df, x='Correlation', y='Law Type', orientation='h',
              color='pos',
              height = 1000,
              title=f'RQ2: Association of Each Law Strength with Firearm Death Rate\n({df["year"].max()} data, n = {len(recent_data)} states)',
              hover_name="Law Type", hover_data={'pos':False, 'Law Type':False, 'Correlation': True}
              )
fig1a.update_layout(xaxis_title='Correlation with Death Rate', showlegend=False)
fig1a.update_yaxes(type='category')            
fig1a.update_yaxes({'gridcolor': 'white'})        
fig1a.update_xaxes({'gridcolor': None, 'zerolinecolor': 'black', 'linecolor': None, 'zerolinewidth': 2})    
fig1a.show()

cor2_df['pos'] = cor2_df['Correlation'] > 0
fig2a = px.bar(cor2_df, x='Correlation', y='Law Type', orientation='h',
            color='pos',
            category_orders={"Law Type": cor2_df["Law Type"].to_list()},
            height = 1000,
            title=f'RQ2: Association of Each Feature with Firearm Death Rate\n({df["year"].max()} data, n = {len(recent_data)} states)',
            hover_name="Law Type", hover_data={'pos':False, 'Law Type':False, 'Correlation': True}
            )
fig2a.update_layout(xaxis_title='Correlation with Death Rate', showlegend=False)
fig2a.update_yaxes(type='category')            
fig2a.update_yaxes({'gridcolor': 'white'})        
fig2a.update_xaxes({'gridcolor': None, 'zerolinecolor': 'black', 'linecolor': None, 'zerolinewidth': 2})    
fig2a.update_layout(xaxis_title='Correlation with Death Rate', yaxis_title='Feature', showlegend=False, yaxis={'dtick': 1})

fig2a.show()

# ### 2.2 Lasso Reduction
# Running lasso reduction against law strenght and 
# feature classes to see which ones reduction identifies as important. 
# Compare 1 degree reductions against correlation results above.


def lasso_reduction(X, y, degrees=[1,2,3]):

    models = []
    for degree in degrees:
        print(f"Fitting Lasso Degree {degree}")
        model = Pipeline([
            ('poly', PolynomialFeatures(degree)), 
            ('scaler', StandardScaler()), 
            ('lasso', Lasso())
            ])
        fit = model.fit(X, y)
        models.append({'degree': degree, 'model': model, 'fit': fit})

    fig = plt.scatter(X.index, y, label=f'Actuals', alpha=0.5)
    plt.xlabel('Row Index')
    plt.ylabel('Predicted y')
    plt.title('Lasso Regression Fits')
    for model in models:
        degree = model['degree']
        fit = model['fit']
        y_plot = fit.predict(X)
        plt.scatter(X.index, y_plot, label=f'Lasso Degree {degree}', alpha=0.5)
    plt.legend()
    plt.show()

    for model in models:
        degree = model['degree']
        fit = model['fit']
        coefficients = fit['lasso'].coef_
        features = fit['poly'].get_feature_names_out()

        # Identify features that were kept (non-zero coefficients)
        # If you had feature names, you could map them
        selected_features_indices = np.where(coefficients != 0)[0]
        removed_features_indices = np.where(coefficients == 0)[0]

        print(f"\nLasso Degree {degree} Results:")
        print(f"\tNumber of features selected: {len(selected_features_indices)}")
        print(f"\tNumber of features removed: {len(removed_features_indices)}")

        # You can access the final sparse model's coefficients
        print("\tCoefficients of selected features:")
        # for i in selected_features_indices:
        #     print(f"\t  {coefficients[i]:10f} '{features[i]}'")

        cf = pd.DataFrame({
            'Coefficient': coefficients[selected_features_indices],
            'Feature': features[selected_features_indices]
            }).sort_values(by='Coefficient', ascending=False)
        with pd.option_context('display.max_rows', None,):  
            print(cf.to_string(
                index=False, justify='left', 
                formatters={'Feature': '{:<200}'.format, 'Coefficient': '{:>10.6f}'.format}
                ))


    return {'models': models}

# Lasso Regression
degrees = [1, 2, 3]

_ = lasso_reduction(noDC[law_features], noDC['rate'], degrees)

# %%
_ = lasso_reduction(noDC[restrictive_classes + permissive_classes], noDC['rate'], degrees)


# ### 3.1 K-Means Clustering
# 
# K-Means partitions states into 
# k clusters by minimizing within-cluster variance.
#  We first determine the optimal number of clusters using the elbow method, 
# then characterize each cluster.

# Filter to most recent year
latest_year = df['year'].max()
df_latest = df[df['year'] == latest_year].copy()

# Features for clustering: all law strength variables + composite score
cluster_features = law_features + ['law_strength_score']
X_cluster = df_latest[cluster_features]

# Standardize features
scaler_cluster = StandardScaler()
X_scaled = scaler_cluster.fit_transform(X_cluster)

print(f"Clustering {len(df_latest)} states using {len(cluster_features)} features")

# Elbow Method to determine optimal k
inertias = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(k_range, inertias, 'o-', linewidth=2, markersize=8)
plt.axvline(x=3, color='red', linestyle='--', label='Suggested k=3')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (Inertia)')
plt.title('K-Means: Elbow Method for Optimal k Selection')
plt.legend()
plt.tight_layout()
plt.show()

print("The elbow appears around k=3, suggesting three distinct state groupings.")

# Fit K-Means with k=3
K = 3
kmeans_final = KMeans(n_clusters=K, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(X_scaled)
df_latest['cluster'] = cluster_labels

# Characterize clusters
cluster_summary = df_latest.groupby('cluster').agg({
    'rate': 'mean',
    'law_strength_score': 'mean',
    'state_name': 'count'
}).rename(columns={'state_name': 'n_states'})

cluster_summary = cluster_summary.sort_values('law_strength_score', ascending=False)

print("K-Means Clustering Results (k=3)")
print("="*60)
print(cluster_summary.round(2))

# Create interpretive labels
cluster_order = cluster_summary.index.tolist()
cluster_names = {
    cluster_order[0]: 'Most Restrictive',
    cluster_order[1]: 'Moderately Restrictive', 
    cluster_order[2]: 'Least Restrictive'
}
df_latest['cluster_name'] = df_latest['cluster'].map(cluster_names)

print("\nCluster Interpretation:")
for cluster_id in cluster_order:
    name = cluster_names[cluster_id]
    rate = cluster_summary.loc[cluster_id, 'rate']
    score = cluster_summary.loc[cluster_id, 'law_strength_score']
    print(f"  Cluster {cluster_id} ({name}): Avg Rate = {rate:.1f}, Avg Law Score = {score:.1f}")


# ### 3.2 Hierarchical Clustering
# 
# Hierarchical clustering builds a tree-like structure (dendrogram) 
# showing how states merge into clusters. 
# This provides an alternative view of state groupings and 
# validates our K-Means results.

# Hierarchical clustering with Ward's method
Z = linkage(X_scaled, method='ward')

plt.figure(figsize=(16, 8))
dendrogram(
    Z,
    labels=df_latest['state_name'].values,
    leaf_rotation=90,
    leaf_font_size=8
)
plt.title('Hierarchical Clustering Dendrogram (Ward Linkage)')
plt.xlabel('States')
plt.ylabel('Euclidean Distance')
plt.tight_layout()
plt.show()

print("The dendrogram shows natural groupings of states based on gun law similarity.")
print("Cutting at height ~20-25 yields approximately 3 clusters, consistent with K-Means.")

# ### 3.3 PCA Visualization of Clusters
# 
# Principal Component Analysis reduces our high-dimensional feature space 
# to 2 dimensions for visualization, 
# allowing us to see how well-separated the clusters are.

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                      c=df_latest['cluster'], cmap='viridis', 
                      s=100, alpha=0.7, edgecolors='black', linewidth=0.5)

# Add state labels
for i, state in enumerate(df_latest['state'].values):
    plt.annotate(state, (X_pca[i, 0], X_pca[i, 1]), fontsize=8, alpha=0.7)

plt.colorbar(scatter, label='Cluster')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
plt.title('PCA Visualization of State Clusters Based on Gun Law Profiles')
plt.tight_layout()
plt.show()

print(f"Total variance explained by first 2 PCs: {sum(pca.explained_variance_ratio_)*100:.1f}%")


# ### 3.4 Supervised Classification: KNN and MLP
# 
# Having identified meaningful clusters, 
# we now train supervised classifiers to predict cluster membership. 
# This demonstrates whether the cluster structure is learnable 
# and could be applied to classify new or hypothetical states.

# Prepare data for supervised classification
y_clusters = df_latest['cluster'].values

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_scaled, y_clusters, test_size=0.3, random_state=42, stratify=y_clusters
)

print(f"Training set: {len(X_train_c)} states")
print(f"Test set: {len(X_test_c)} states")

# KNN Classifier
knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train_c, y_train_c)
y_pred_knn = knn_clf.predict(X_test_c)

acc_knn = accuracy_score(y_test_c, y_pred_knn)

print("KNN Classifier (k=5)")
print("="*50)
print(f"Test Accuracy: {acc_knn:.2%}")
print("\nClassification Report:")
target_names = ['Least Restrictive', 'Most Restrictive', 'Moderate']
print(classification_report(y_test_c, y_pred_knn, target_names=target_names, zero_division=0))

# MLP Classifier
mlp_clf = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
)
mlp_clf.fit(X_train_c, y_train_c)
y_pred_mlp = mlp_clf.predict(X_test_c)

acc_mlp = accuracy_score(y_test_c, y_pred_mlp)

print("MLP Classifier (100-50 hidden layers)")
print("="*50)
print(f"Test Accuracy: {acc_mlp:.2%}")
print("\nClassification Report:")
print(classification_report(y_test_c, y_pred_mlp, target_names=target_names, zero_division=0))

# Model comparison
print("\nClassification Model Comparison")
print("="*40)
print(f"KNN (k=5):         {acc_knn:.2%} accuracy")
print(f"MLP (100-50):      {acc_mlp:.2%} accuracy")
print(f"\n★ Best Classifier: {'MLP' if acc_mlp > acc_knn else 'KNN'}")


# ### 3.5 Demonstration: Classifying a Hypothetical State
# 
# To demonstrate the practical utility of our classifiers, 
# we create a hypothetical state with a moderate gun law profile 
# and predict which cluster it would belong to.

# Create a hypothetical moderate state
hypothetical = pd.DataFrame({
    'strength_background_checks': [7],
    'strength_carrying_a_concealed_weapon_ccw': [-1],
    'strength_castle_doctrine': [-2],
    'strength_dealer_license': [1],
    'strength_firearm_sales_restrictions': [4],
    'strength_local_laws_preempted_by_state': [0],
    'strength_minimum_age': [6],
    'strength_prohibited_possessor': [5],
    'strength_registration': [0],
    'strength_waiting_period': [2],
    'strength_firearm_removal_at_scene_of_domestic_violence': [1],
    'strength_firearms_in_college_university': [0],
    'strength_child_access_laws': [1],
    'strength_gun_trafficking': [1],
    'strength_open_carry': [0],
    'strength_required_reporting_of_lost_or_stolen_firearms': [1],
    'strength_safety_training_required': [1],
    'strength_untraceable_firearms': [1],
    'strength_permit_to_purchase': [1],
    'strength_firearms_in_k_12_educational_settings': [0],
    'law_strength_score': [35]
})

# Scale using the same scaler
hyp_scaled = scaler_cluster.transform(hypothetical)

# Predictions
pred_knn_hyp = knn_clf.predict(hyp_scaled)[0]
pred_mlp_hyp = mlp_clf.predict(hyp_scaled)[0]
prob_mlp = mlp_clf.predict_proba(hyp_scaled)[0]

print("Hypothetical State Classification")
print("="*50)
print(f"Input: Composite law strength score = 35 (moderate)")
print(f"\nKNN Prediction: Cluster {pred_knn_hyp} ({cluster_names.get(pred_knn_hyp, 'Unknown')})")
print(f"MLP Prediction: Cluster {pred_mlp_hyp} ({cluster_names.get(pred_mlp_hyp, 'Unknown')})")
print(f"\nMLP Confidence (probability per cluster):")
for i, prob in enumerate(prob_mlp):
    print(f"  Cluster {i}: {prob:.1%}")


# 
# ## Research Question 4: Temporal Trends
# 
# **Question**: How have the relationships between gun laws
#  and death rates evolved over time?
# 
# We examine national trends and state-level trajectories from 2014 to 2023.

# National trends
national = df.groupby('year').agg({
    'rate': 'mean',
    'law_strength_score': 'mean'
}).round(2)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(national.index, national['rate'], 'o-', color='darkred', linewidth=2, markersize=8)
axes[0].set_xlabel('Year')
axes[0].set_ylabel('Average Death Rate per 100k')
axes[0].set_title('National Average Firearm Death Rate Over Time')
axes[0].grid(True, alpha=0.3)

axes[1].plot(national.index, national['law_strength_score'], 'o-', color='steelblue', linewidth=2, markersize=8)
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Average Law Strength Score')
axes[1].set_title('National Average Law Strength Over Time')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

rate_change = national['rate'].iloc[-1] - national['rate'].iloc[0]
law_change = national['law_strength_score'].iloc[-1] - national['law_strength_score'].iloc[0]
print(f"Change from {df['year'].min()} to {df['year'].max()}:")
print(f"  Death Rate: {rate_change:+.2f} per 100k")
print(f"  Law Strength: {law_change:+.2f} points")


# State-level changes
changes = df[df['year'].isin([df['year'].min(), df['year'].max()])].pivot_table(
    index='state_name', columns='year', values=['rate', 'law_strength_score']
).dropna()

changes.columns = ['_'.join(map(str, col)) for col in changes.columns]
min_yr, max_yr = df['year'].min(), df['year'].max()
changes['rate_change'] = changes[f'rate_{max_yr}'] - changes[f'rate_{min_yr}']
changes['law_change'] = changes[f'law_strength_score_{max_yr}'] - changes[f'law_strength_score_{min_yr}']

# Correlation
r, p = pearsonr(changes['law_change'], changes['rate_change'])

plt.figure(figsize=(12, 8))
plt.scatter(changes['law_change'], changes['rate_change'], s=80, alpha=0.6, edgecolors='black')

# Trend line
z = np.polyfit(changes['law_change'], changes['rate_change'], 1)
p_line = np.poly1d(z)
x_line = np.linspace(changes['law_change'].min(), changes['law_change'].max(), 100)
plt.plot(x_line, p_line(x_line), 'r-', linewidth=2, alpha=0.8, label=f'Trend (r={r:.3f})')

plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

plt.xlabel('Change in Law Strength Score')
plt.ylabel('Change in Death Rate per 100k')
plt.title(f'State-Level Changes: Law Strength vs. Death Rate ({min_yr}–{max_yr})')
plt.legend()
plt.tight_layout()
plt.show()

print(f"Correlation: r = {r:.3f}, p = {p:.4f}")
if r < 0:
    print("States that strengthened gun laws tended to see decreases in death rates.")
else:
    print("No consistent relationship between law changes and rate changes.")

