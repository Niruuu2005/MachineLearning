import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv("D:/mine/Programs/5thsem/ml/Iris.csv")
print("Orignal DataFrame:\n",df)

preprocessing_steps = [
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()), 
]
preprocessor = Pipeline(preprocessing_steps)

numericalDF = df.select_dtypes(include=['number'])

arr = []
for i in numericalDF:
    arr.append(i)

preprocessingDF = pd.DataFrame(preprocessor.fit_transform(numericalDF), columns=arr)

for i in arr: df[i] = preprocessingDF[i]
    
print ("\nDataFrame before Reduction:\n",df)

n_components = 2
pca = PCA(n_components)
reduced = pca.fit_transform(pd.DataFrame(df[arr]))
reduced_df = pd.DataFrame(data=reduced, columns=[f'PC{i}' for i in range(1, n_components+1)])
df = pd.concat([reduced_df, df.select_dtypes(exclude=['number'])], axis=1)

print ("\nDataFrame after Reduction:\n",df)
