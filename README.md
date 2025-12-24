## EXNO-3-DS

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
      *****FEATURE ENCODING*****

 import pandas as pd
 
 df=pd.read_csv("Encoding Data.csv")
 df
 
 output<img width="676" height="379" alt="image" src="https://github.com/user-attachments/assets/6498ac80-766a-46ce-9334-b448bcd11ea8" />

from sklearn.preprocessing import LabelEncoder,OrdinalEncoder

pm=['Hot','Warm','Cold']

e1=OrdinalEncoder(categories=[pm])

e1.fit_transform(df[["ord_2"]])

output<img width="370" height="222" alt="image" src="https://github.com/user-attachments/assets/115901e8-f0fa-4891-b73b-f2a860c3fb06" />

df['bo2']=e1.fit_transform(df[["ord_2"]])
df

output<img width="658" height="380" alt="image" src="https://github.com/user-attachments/assets/c04e009a-f305-4c69-9856-f37d113e7386" />

le=LabelEncoder()

dfc=df.copy()

dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc

output<img width="590" height="379" alt="image" src="https://github.com/user-attachments/assets/f116a79c-a017-4760-ac4b-d2d7c40c096f" />

from sklearn.preprocessing import OneHotEncoder

ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()

enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))

df2=pd.concat([df2,enc],axis=1)
df2

output<img width="699" height="384" alt="image" src="https://github.com/user-attachments/assets/c576357b-e3ea-4c47-94cf-04ee43d6b5b5" />

pd.get_dummies(df2,columns=["nom_0"])

output<img width="850" height="387" alt="image" src="https://github.com/user-attachments/assets/b2d5065e-6b08-4fe0-89fc-107518d2fffe" />

pip install --upgrade category_encoders
<img width="1260" height="510" alt="image" src="https://github.com/user-attachments/assets/0c8074d0-055c-4b17-92a3-b37715e95295" />

from category_encoders import BinaryEncoder

df=pd.read_csv("data.csv")
df

output<img width="674" height="376" alt="image" src="https://github.com/user-attachments/assets/7a34e85b-b790-4515-9e61-c756180de127" />

be=BinaryEncoder()

nd=be.fit_transform(df['Ord_2'])

dfb=pd.concat([df,nd],axis=1)

dfb1=df.copy()
dfb

output<img width="968" height="383" alt="image" src="https://github.com/user-attachments/assets/c908a37d-e706-4d9c-9fe5-eabb4323f1ae" />

from category_encoders import TargetEncoder

te=TargetEncoder()

cc=df.copy()

new=te.fit_transform(X=cc["City"],y=cc["Target"])

cc=pd.concat([cc,new],axis=1)
cc

output<img width="819" height="395" alt="image" src="https://github.com/user-attachments/assets/0f76f73b-2615-42ea-9f42-e80e6bbc7625" />


     *****FEATURE TRANSFORMATION*****

import pandas as pd
from scipy import stats
import numpy as np

df=pd.read_csv("Data_to_Transform.csv")
df

output<img width="1048" height="458" alt="image" src="https://github.com/user-attachments/assets/eb4f06ea-3f94-4661-b419-f68d0d45fa86" />

df.skew()

output<img width="813" height="118" alt="image" src="https://github.com/user-attachments/assets/262d30a3-0ae9-4c0b-aea5-95edca00bec4" />

np.log(df["Highly Positive Skew"])

output<img width="798" height="273" alt="image" src="https://github.com/user-attachments/assets/a3996fc7-42e8-4c27-a4af-549d041d59c5" />

np.reciprocal(df["Moderate Positive Skew"])

output<img width="824" height="278" alt="image" src="https://github.com/user-attachments/assets/3ec24cd2-3519-4b42-9088-7317096c97f4" />

np.sqrt(df["Highly Positive Skew"])

output<img width="957" height="278" alt="image" src="https://github.com/user-attachments/assets/7fb2a534-ee35-4bed-894f-2e4cb9201517" />

np.square(df["Highly Positive Skew"])

output<img width="831" height="268" alt="image" src="https://github.com/user-attachments/assets/c07a2c41-bb50-4563-b2b9-cb768078e091" />

df["Highly Positive Skew_boxcox"],parammeters=stats.boxcox(df["Highly Positive Skew"])
df

output<img width="1278" height="460" alt="image" src="https://github.com/user-attachments/assets/89d342c6-484e-4799-8a62-1fb54230d46d" />

df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"]
df.skew()

output<img width="691" height="178" alt="image" src="https://github.com/user-attachments/assets/c6e13e4b-05fe-4e0b-80e7-ed171ebf2de9" />

df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()

output<img width="855" height="200" alt="image" src="https://github.com/user-attachments/assets/4b3ebc94-46af-41c8-b026-319a50a1337a" />

from sklearn.preprocessing import QuantileTransformer

qt=QuantileTransformer(output_distribution='normal')

df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df

output<img width="1342" height="486" alt="image" src="https://github.com/user-attachments/assets/08287402-8577-4061-8083-a6da52d244c1" />

import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()

output<img width="1053" height="562" alt="image" src="https://github.com/user-attachments/assets/29486f6c-73de-45cc-b128-77ab2dc65a44" />

sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()

output<img width="1057" height="560" alt="image" src="https://github.com/user-attachments/assets/28544497-e46c-4e59-be65-e95a427d9bb5" />

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()

output<img width="1047" height="557" alt="image" src="https://github.com/user-attachments/assets/0e7dd346-2f6a-4d77-a113-44ccc5a24c16" />

df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df['Highly Negative Skew'],line='45')
plt.show()

output<img width="1080" height="554" alt="image" src="https://github.com/user-attachments/assets/29e25df9-c0d4-4e3f-8f86-b2d44d49cdfb" />

sm.qqplot(df['Highly Negative Skew_1'],line='45')
plt.show()

output<img width="1034" height="543" alt="image" src="https://github.com/user-attachments/assets/bc642b5a-abac-4373-81bb-a6603e5b67ae" />

dt=pd.read_csv("titanic_dataset.csv")

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

dt["Age_1"]=qt.fit_transform(dt[["Age"]])

sm.qqplot(dt['Age'],line='45')
plt.show()

output<img width="1116" height="553" alt="image" src="https://github.com/user-attachments/assets/0af84291-cbb7-4df4-a439-19c57bf584b7" />

sm.qqplot(dt['Age_1'],line='45')
plt.show()

output<img width="1050" height="547" alt="image" src="https://github.com/user-attachments/assets/4c5dfdad-5092-4413-8ca9-f260b853d895" />

# RESULT:

 Thus the program to implement the linear regression using gradient descent is written and verified using python programming. 


       
