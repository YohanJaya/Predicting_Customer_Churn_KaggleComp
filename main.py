import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gplearn.genetic import SymbolicTransformer
from sklearn.feature_selection import mutual_info_regression
import seaborn as sns
import matplotlib.pyplot as plt

dfTrain = pd.read_csv('/kaggle/input/competitions/playground-series-s6e3/train.csv')
dfTest = pd.read_csv('/kaggle/input/competitions/playground-series-s6e3/test.csv')
dfTrain.shape
dfTrain.head()
dfTrain.info()
dfTrain.describe()


#plotting graphs for each feature

# Numerical features
num_cols = dfTrain.select_dtypes(include=['int64','float64']).columns

for col in num_cols:
    plt.figure()
    sns.histplot(dfTrain[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.show()


# Categorical features
cat_cols = dfTrain.select_dtypes(include=['object','category']).columns

for col in cat_cols:
    plt.figure()
    sns.countplot(x=dfTrain[col])
    plt.title(f'Count Plot of {col}')
    plt.xlabel(col)
    plt.show()
    
    
#converting the data type of the SineorCotizens to object type
dfTrain['SeniorCitizen'] = dfTrain['SeniorCitizen'].astype('object')

#detecting outliers
for col in dfTrain.columns:
    if dfTrain[col].dtype == 'int64' or dfTrain[col].dtype == 'float64':
        q1 = dfTrain[col].quantile(0.25)
        q3 = dfTrain[col].quantile(0.75)

        lower = q1 - 1.5 *(q3 - q1)
        upper = q3 + 1.5 * (q3 - q1)

    
        outliers = dfTrain[(dfTrain[col] < lower) | (dfTrain[col] > upper)][col]

        percentage = (len(outliers) / len(dfTrain[col])) * 100

        print(col, "Outlier %:", percentage)
        
#Looking what sort of elements are in categorical features
for col in dfTrain.columns:
    if dfTrain[col].dtype != 'int64' and dfTrain[col].dtype != 'float64':
        print(f'{col} : {dfTrain[col].unique()}')
        
#replace Yes, No and other catergories with numerical values
binCat= ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling','Churn']
mapping = {'Yes' : 1, 'No' : 0}

for col in binCat:
    if col != 'Churn' :
        dfTest[col] = dfTest[col].map(mapping)

    dfTrain[col] = dfTrain[col].map(mapping)

#one hot encoding 
dfTrain = pd.get_dummies(dfTrain)



def mi_metric(y, y_pred, **kwargs):
    # mutual_info_regression expects 2D X
    y_pred = y_pred.reshape(-1, 1)

    # handle constant predictions (MI fails sometimes)
    # flatten() >> make 1D array from a high dimension array
    # set() >> make a set, means takes only the unique values
    
    if len(set(y_pred.flatten())) <= 1:
        return 0
        
    mi = mutual_info_regression(y_pred, y)
    # gplearn maximizes the metric, so return MI
    return mi[0]

# Initialize SymbolicTransformer with custom metric
transformer_mi = SymbolicTransformer(
    generations=10,
    population_size=100,
    hall_of_fame=5,
    n_components=3,
    function_set=('add','sub','mul','div','sqrt','log','abs'),
    metric=mi_metric,  # custom MI metric
    verbose=1,
    random_state=42
)




