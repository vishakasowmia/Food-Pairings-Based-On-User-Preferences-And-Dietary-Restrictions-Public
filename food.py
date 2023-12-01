

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import FunctionTransformer

"""#Reading the Dataset"""

df=pd.read_csv("dataset.csv")# Creating the Dataframe to read the file
# df

df.info()#Info returns the basic info of count and dtypes

df = df.drop("Folate/µg",axis=1)# Drop the column Folate/ug

# df.tail()# Read last 5 defaults

# df.head() # Read first 5 defaults

# Replace the column names with the actual column names from your data
columns = ["title", "vegetarian", "vegan", "glutenFree", "dairyFree", "ingredients", "sustainable", "veryHealthy",
           "lowFodmap", "ketogenic", "healthScore", "percentProtein", "percentFat", "percentCarbs", "calories",
           "Fat/g", "Vitamin E/mg", "Carbohydrates/g", "Magnesium/mg", "Cholesterol/mg", "Zinc/mg", "Sodium/mg", "Copper/mg", "Sugar/g", "Fiber/g", "Alcohol/g"]

# Create a DataFrame
df = pd.DataFrame(df, columns=columns)

# Calculate median value for Fiber/g
median_fiber = df['Fiber/g'].median()

# Calculate average values for percentProtein, percentFat, and percentCarbs
avg_percent_protein = df['percentProtein'].mean()
avg_percent_fat = df['percentFat'].mean()
avg_percent_carbs = df['percentCarbs'].mean()

# Create the new attributes based on the specified criteria
df['Diabetes'] = df['Fiber/g'].apply(lambda x: 1 if x > median_fiber else 0)
df['High Protein'] = df['percentProtein'].apply(lambda x: 1 if x > avg_percent_protein else 0)
df['High Fat'] = df['percentFat'].apply(lambda x: 1 if x > avg_percent_fat else 0)
df['Low Carbohydrate'] = df['percentCarbs'].apply(lambda x: 1 if x < avg_percent_carbs else 0)

# Display the modified DataFrame
# print(df)

# Calculate average values for percentProtein, percentFat, and percentCarbs
avg_percent_protein = df['percentProtein'].mean()
avg_percent_fat = df['percentFat'].mean()
avg_percent_carbs = df['percentCarbs'].mean()
avg_Fiber_g = df['Fiber/g'].mean()
# Create the "Diabetes" column based on the specified criteria
df['Diabetes'] = df.apply(lambda row: 1 if #row['percentProtein'] > avg_percent_protein and
                                          #row['percentFat'] > avg_percent_fat and
                                          row['percentCarbs'] < avg_percent_carbs
                                          #row['Fiber/g'] > avg_Fiber_g
                           else 0, axis=1)

# Display the modified DataFrame with the "Diabetes" column
# print(df[['Diabetes']])

# df.head()

unique_values_sum = df["Diabetes"].value_counts()# Values count
unique_values_sum

mean_percent_protein = df['percentProtein'].mean()
mean_percent_fat = df['percentFat'].mean()
mean_cholesterol = df['Cholesterol/mg'].mean()

# Create the "Obesity" column based on the specified criteria
df['Obesity'] = df.apply(lambda row: 1 if row['percentFat'] < mean_percent_fat
                         else 0, axis=1)

# Display the modified DataFrame with the "Obesity" column
# print(df[['Obesity']])

unique_value = df["Obesity"].value_counts()# value counts
# unique_value

# print(df)

mean_sodium = df['Sodium/mg'].mean()
# Create the "Obesity" column based on the specified criteria
df['BP'] = df.apply(lambda row: 1 if row['Sodium/mg'] < mean_sodium
                         else 0, axis=1)
# Display the modified DataFrame with the "Obesity" column
# print(df[['title', 'BP']])

unique_value = df["BP"].value_counts()# Value counts
# unique_value

# print(df)

df.Diabetes.value_counts()# diabetes value counts

df.head(30)# Read first 30 by defaults

mean_cholorestrol = df['Cholesterol/mg'].mean()
mean_fat =  df['Fat/g'].mean()
mean_sodium = df['Sodium/mg'].mean()
# Create the "Obesity" column based on the specified criteria
df['Heart'] = df.apply(lambda row: 1 if #row['Cholesterol/mg'] > mean_cholorestrol and
                                        #row['Fat/g'] > mean_fat and
                                        row['Sodium/mg'] < mean_sodium
                         else 0, axis=1)

# Display the modified DataFrame with the "Obesity" column
# print(df[['Heart']])

unique_value = df["Heart"].value_counts()
# print(unique_value)

# print(df)

max_calories=16510860
max_daily_fat=16510860
max_daily_Carbohydrate=24242.450000
max_daily_Fiber=10375.010000
max_daily_Sugar=5808.020000
max_daily_Protein=82.390000
max_list=[max_calories,max_daily_fat,max_daily_Carbohydrate,max_daily_Fiber,max_daily_Sugar,max_daily_Fiber,max_daily_Protein]

extracted_data=df.copy()
for column,maximum in zip(extracted_data.columns[10:27],max_list):
    extracted_data=extracted_data[extracted_data[column]<maximum]

extracted_data.iloc[:,10:33].corr()

from sklearn.preprocessing import StandardScaler# Standard Scaler and fit it and transform
scaler=StandardScaler()
prep_data=scaler.fit_transform(extracted_data.iloc[:,10:27].to_numpy())

# Prep Data after standard scaler

# print(prep_data)

"""
Cosine similarity in machine learning can be used as a metric for deciding the optimal number of neighbors 
where the data points with a higher similarity will be considered as the nearest neighbors 
and the data points with lower similarity will not be considered.
"""


neigh = NearestNeighbors(metric='cosine',algorithm='brute')
neigh.fit(prep_data)

transformer = FunctionTransformer(neigh.kneighbors,kw_args={'return_distance':False})
pipeline=Pipeline([('std_scaler',scaler),('NN',transformer)])

params={'n_neighbors':10,'return_distance':False}
pipeline.get_params()
pipeline.set_params(NN__kw_args=params)

pipeline.transform(extracted_data.iloc[0:1,10:27].to_numpy())[0]

extracted_data.iloc[pipeline.transform(extracted_data.iloc[0:1,10:27].to_numpy())[0]]

# Taking user input from user to select vergetarian and Non-Vegetarian
# food_type = input("Enter your preference: (Vegetarian, Non-Vegetarian)").lower()


def generate_recommendation(food_type, disease_choice) : 
  if food_type == "vegetarian":
  # And Check with disease choice for the obesity, BP and Heart Disease and return the result of selected columns
      if disease_choice.lower() in ["obesity", "bp", "heart"]:
          if disease_choice.upper() == "BP":
            #input_value = int(input(f"Enter the {disease_choice.upper()} value (0 or 1): "))
            filtered_data = extracted_data[
              (extracted_data["vegetarian"] == True) & (extracted_data[disease_choice.upper()] == 1)
            ]
            filtered_data.reset_index(inplace=True)
            if not filtered_data.empty:
              output = []
              for i in range(len(filtered_data)):
                output.append(filtered_data.loc[i,"title"])
              return output
            else:
              return "No matching food items found."
          else:
            #input_value = int(input(f"Enter the {disease_choice.upper()} value (0 or 1): "))
            print(disease_choice.capitalize())
            filtered_data = extracted_data[
              (extracted_data["vegetarian"] == True) & (extracted_data[disease_choice.capitalize()] == 1)
            ]
            filtered_data.reset_index(inplace=True)
            if not filtered_data.empty:
              output = []
              for i in range(len(filtered_data)):
                output.append(filtered_data.loc[i,"title"])
              return output
            else:
              return "No matching food items found."
      else:
          return "Invalid disease choice."
  elif food_type == "non-vegetarian":
      # disease_choice = input("Choose a disease (obesity, BP, Heart): ")
      if disease_choice.lower() in ["obesity", "bp", "heart"]:
          if disease_choice.upper() == "BP":
            #input_value = int(input(f"Enter the {disease_choice.upper()} value (0 or 1): "))
            filtered_data = extracted_data[
              (extracted_data["vegetarian"] == 0) & (extracted_data[disease_choice.upper()] == 1)
            ]
            filtered_data.reset_index(inplace=True)
            if not filtered_data.empty:
              output = []
              for i in range(len(filtered_data)):
                output.append(filtered_data.loc[i,"title"])
              return output
            else:
              return "No matching food items found."
          else:
            #input_value = int(input(f"Enter the {disease_choice.upper()} value (0 or 1): "))
            # print(disease_choice.capitalize())
            filtered_data = extracted_data[
              (extracted_data["vegetarian"] == True) & (extracted_data[disease_choice.capitalize()] == 1)
            ]
            filtered_data.reset_index(inplace=True)
            if not filtered_data.empty:
              output = []
              for i in range(len(filtered_data)):
                output.append(filtered_data.loc[i,"title"])
              return output
            else:
              return "No matching food items found."
      else:
          return "Invalid disease choice."
