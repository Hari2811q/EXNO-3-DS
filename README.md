## EXNO-3-DS
```
Developed by : Hariprasath R
Reg. No : 212223040059
```

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```
<img width="631" height="459" alt="image" src="https://github.com/user-attachments/assets/9d4a5f0f-3b33-4d7e-b458-b8a068bd9d99" />

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
<img width="346" height="240" alt="image" src="https://github.com/user-attachments/assets/84eb073a-b1fc-45a7-b66a-7aaa8c307023" />

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
<img width="555" height="456" alt="image" src="https://github.com/user-attachments/assets/2cd3ab94-80d0-4e08-9518-2c3eddc01e72" />

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
<img width="717" height="508" alt="image" src="https://github.com/user-attachments/assets/fd4a325e-3ab8-4733-aac4-c419cafc9371" />

~~~
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df
~~~
<img width="560" height="456" alt="image" src="https://github.com/user-attachments/assets/4d9a3aca-8cc5-49ca-bf5e-cc0c6ae99caf" />

~~~
pd.get_dummies(df2,columns=["nom_0"])
~~~
<img width="898" height="456" alt="image" src="https://github.com/user-attachments/assets/b4709d51-710c-46a0-8308-e618a3267b60" />

~~~
pip install --upgrade category_encoders
~~~
<img width="1509" height="447" alt="image" src="https://github.com/user-attachments/assets/fa1286b8-94a7-4329-a161-3f85ec43e211" />

~~~
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
dfb=pd.concat([df,nd],axis=1)
dfb
~~~
<img width="1007" height="458" alt="image" src="https://github.com/user-attachments/assets/248540c1-31ff-4e5e-a45a-09c2049ce51c" />

~~~
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
~~~
<img width="775" height="457" alt="image" src="https://github.com/user-attachments/assets/5972af14-f254-4cfb-9846-0807116c6d19" />

~~~
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
~~~
<img width="1146" height="532" alt="image" src="https://github.com/user-attachments/assets/ac45a9d7-7d13-459d-aeb4-6ffd71f33025" />

~~~
df.skew()
~~~
<img width="452" height="258" alt="image" src="https://github.com/user-attachments/assets/f5767c27-c61d-4f26-91e2-a6f30f1b77ad" />

~~~
np.log(df["Highly Positive Skew"])
~~~
<img width="457" height="571" alt="image" src="https://github.com/user-attachments/assets/86389cd6-f4f5-4b6d-aae6-9824fcddfa20" />

~~~
np.reciprocal(df["Moderate Positive Skew"])
~~~
<img width="542" height="574" alt="image" src="https://github.com/user-attachments/assets/6a06f744-dfd6-47c4-86cd-143bee35bbcc" />

~~~
np.sqrt(df["Highly Positive Skew"])
~~~
<img width="563" height="576" alt="image" src="https://github.com/user-attachments/assets/bff5358b-8710-4ecf-a8f3-9b81669eb3b4" />

~~~
np.square(df["Highly Positive Skew"])
~~~
<img width="491" height="570" alt="image" src="https://github.com/user-attachments/assets/08a66891-a4d5-4dc4-8178-c48528ea3f34" />

~~~
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
~~~
<img width="1345" height="525" alt="image" src="https://github.com/user-attachments/assets/12a0acc0-9ad6-4d70-af07-618457dcfd13" />

~~~
df.skew()
~~~
<img width="498" height="301" alt="image" src="https://github.com/user-attachments/assets/d3dbc3e1-df05-46d9-b5cd-2ab6df8f1322" />

~~~
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
~~~
<img width="570" height="345" alt="image" src="https://github.com/user-attachments/assets/48fdb8ae-65e8-4263-9f38-0e44856724b9" />

~~~
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
~~~
<img width="1510" height="554" alt="image" src="https://github.com/user-attachments/assets/9383e698-23d4-43df-8056-7a5cd2b14657" />

~~~
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
~~~
<img width="856" height="566" alt="image" src="https://github.com/user-attachments/assets/53d93d7c-8e6b-4d87-a826-74330832ad83" />

~~~
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
~~~
<img width="836" height="556" alt="image" src="https://github.com/user-attachments/assets/59ea9af7-5cc2-4d1a-867b-8f72816b7a64" />

~~~
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
~~~
<img width="827" height="561" alt="image" src="https://github.com/user-attachments/assets/dcf53a17-2f45-445c-8330-d6d204932f2c" />

~~~
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
~~~
<img width="828" height="556" alt="image" src="https://github.com/user-attachments/assets/04b58559-1623-4f59-8de4-e60f3b419522" />

~~~
dt=pd.read_csv("titanic_dataset.csv")
dt
~~~
<img width="1511" height="572" alt="image" src="https://github.com/user-attachments/assets/43b16a26-00b5-4089-a516-b99404082128" />

~~~
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45') 
plt.show()
~~~
<img width="805" height="557" alt="image" src="https://github.com/user-attachments/assets/4d17a067-7cd3-4eca-9fe4-eff7254fbd3d" />

~~~
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
~~~
<img width="853" height="558" alt="image" src="https://github.com/user-attachments/assets/42158eed-7e23-48ab-9dd8-1d046b3d0a27" />


# RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file
was performed successfully.
       

