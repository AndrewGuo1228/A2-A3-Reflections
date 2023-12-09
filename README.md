# A2 & A3 Reflections

### Author: A heartbroken student from 2023 winter quarter
### Warnings: **Please do remember to submit your assignments and discussion labs after validation!!!** The author of this website forget to submit 16% of overall grade, and he was desparately painful while coding this website, so please do not commmit the same careless and silly mistakes as him did. Besides the content in COGS 108, this website will also involve some important concepts covered in DSC80, serving as a bridge to connect two courses to enhance memory. Hope you all can avoid these unfortunate things, end up with a descent grade, and be benefited from the information provided in this website. **you are welcomed to enrich this websites.** 

---
## Introduction
This website will majorly reflect on what we have learned from assignment 2 and assignment 3, and also point out some potential direction for deeper and further investigation based on each question's context.


# A2

### Q1: Import (0.25 points)
This part teaches us how to import essential packages like pandas, numpy, eaborn, and matplotlib.pyplot.

> Code used:
```
import pandas as pd # used to import pandas (Dataframe)
import numpy as np # used for some common math operations
import seaborn as sns # used for plotting the diagram
import matplotlib.pyplot as plt # used for plotting diagram
```

**reflections and adds on:**

The thing worth noticing in this part of question would be the difference between matplotlib and seaborn, as they might looks similar to the starters, I have done some research and summarize the differences below:


Matplotlib.pyplot: This is a lower-level plotting library that provides a lot of flexibility and control over every aspect of a plot. It's great for creating complex and customized plots, but can require more code for advanced visualizations.

Seaborn (sns): Built on top of Matplotlib, Seaborn is a higher-level interface that is more user-friendly and makes creating attractive and informative statistical graphics easier. It provides a variety of plotting functions that handle many common data visualization needs automatically, like creating plots with default styles and color palettes that are statistically and aesthetically richer.

**In short, Matplotlib.pyplot is a lower-level plotting library that can handle almost all basic plotting needs but if we wants more fancy data visualizations and have more advanced plotting demands, Seaborn would be a better fit (for example, I remember we have used pairplot for this package.)**



### Q2: Read Data into Python (0.25 points) 
This question focused on reading data into Python as a DataFrame using pandas, specifically from a CSV file.

> Code used:
```
pd.read_csv( ‘data path’ ) # read a csv file from given data path
```

**reflections and adds on:**

This question focused specifically on CSV files, but that’s not always the format we are going to get the data from. Therefore, I have done some researches and summarized the way to read other formats of data below:

- Reading JSON Files: json_df = pd.read_json('path_to_your_file.json')
- Reading Excel Files: excel_df = pd.read_excel('path_to_your_file.xlsx')
- Reading from SQL database:
	- import sqlalchemy as sa
	- engine = sa.create_engine('your_database_connection_string')
	- sql_df = pd.read_sql_query('SELECT * FROM your_table', con=engine)



### Q3: Data Summary (1 point)
Here, we learned how to summarize datasets using methods like shape, describe, and generating histograms.

> Code used:
```
df.shape # getting the number of rows and columns
df.describe() # get the statistics for the columns (median, mean, max, min, etc)
Histogram plot: # get the distribution of the data (we can set ‘density’ to True/False)
f1 = plt.gcf()
plt.hist(df['winpercent'], bins=15, edgecolor='black')
plt.xlabel('winpercent')
plt.ylabel('frequency')
plt.title('histogram of winpercent')
```

**reflections and adds on:**

Beyond basic data summaries, we can consider exploring data types and unique values in each column. Investigate the presence of outliers using box plots or scatter plots and ponder their impact on your analysis. Learn about data normalization and standardization techniques, which are crucial when dealing with features of different scales.

For example, in general, we can 
- detect outliers with df.plot(kind='box'). 
- Normalize data: df['column'] = (df['column'] - df['column'].mean()) / df['column'].std().


### Q4: Replace (0.5 points)
This taught us to use dynamic value replacement in datasets using the replace method.

> Code used:
```
df_bool = df.replace(0, False).replace(1, True)
```

**Reflections and adds on:**

We can explore various methods of dealing with missing data: imputation, dropping rows/columns, or using algorithms that can handle missing values. Reflect on the biases that different imputation methods might introduce. Additionally, consider the use of domain knowledge for more intelligent data replacement.

Here's the code:
 - For missing data, explore imputation: df['column'].fillna(df['column'].mean(), inplace=True)
	(Please do note that using mean to perform the imputations is subjected to ruin the distribution of data (data will be clustered around the mean, introducing more randomness while doing the imputation for missing values might give more meaningful informations for later analysis) 
 - drop missing data with df.dropna(inplace=True).
 - Replace based on a condition: df.loc[df['column'] > value, 'column'] = new_value.



### Q5: Barplot (0.4 points) 
in this question, we learned to create and interpret bar plots using seaborn.

> Code used:
```
f2 = plt.gcf()
chocolate_series = df['chocolate']
value_counts = chocolate_series.value_counts()
plt.bar(value_counts.index.astype(str), value_counts.values)
plt.xlabel('Value')
plt.ylabel('Count')
plt.title('Number of True and False Values in Chocolate Series')
plt.xticks([0, 1], ['False', 'True'])
plt.show()
```

**Reflection and adds on:**

We can examine how different aspects of a bar plot, like horizontal vs. vertical bars, can affect the interpretation of data. We can experiment with stacked bar plots or grouped bar plots to display multiple categories of data. Consider how to best convey information in a limited space, like the use of annotations or interactive elements.

Here's the general code:
- Create a vertical bar plot: sns.barplot(x='category_column', y='value_column', data=df).
- For a horizontal bar plot: sns.barplot(x='value_column', y='category_column', data=df).


### Q6: Column Operations (2.25 points) 
This part covered selecting, renaming, and adding new columns in DataFrames.

> Code used:
```
df = df[['competitorname', 'chocolate', 'fruity', 'hard', 'bar', 'pluribus', 'sugarpercent', 'pricepercent', 'winpercent']]
df_bool = df_bool.select_dtypes(include=['object', 'bool'])
df = df.rename({'pluribus': 'multicandy_pack'}, axis='columns')
df_bool = df_bool.rename({'pluribus': 'multicandy_pack'}, axis='columns')
df = df.assign(fruity_choco = df['chocolate'] + df['fruity'])
```

**Reflection and adds on:**

We can dive into advanced column manipulation like converting data types, handling categorical data through one-hot encoding or label encoding, and exploring pandas methods for string manipulation. Reflect on the importance of these operations in preparing data for machine learning models.

Here's the general code:
- Convert data types: df['column'] = df['column'].astype('new_type').
- One-hot encoding: pd.get_dummies(df, columns=['categorical_column']).
	(For the one hot encoding, we can also use Sklearn, the code is provided below as well)
	- from sklearn.preprocessing import OneHotEncoder
	- One_hot = OneHotEncoder( )
	- Pipeline object can then be used to include this Onehotencoder for the categorical data


### Q7: Row Operations (0.9 points) 
We have explored filtering and slicing rows in DataFrames based on specific conditions.

> Code used:
```
both = df.loc[df['fruity_choco'] == 2]
candy_name = both['competitorname'].values[0]
df = df[(df['fruity'] == 1) | (df['chocolate'] == 1)]
```

**Reflection and adds on:**

We can explore more complex row operations like filtering based on multiple conditions, using lambda functions for row-wise operations, and understanding the indexing in pandas.

here's the general code: 
- Filter rows: df_filtered = df[(df['column1'] > value1) & (df['column2'] < value2)].
- Apply a lambda function: df['new_column'] = df['column'].apply(lambda x: function_of_x)



### Q8: Arrange Rows (1.2 points) 
The focus was on sorting DataFrames using sort_values and resetting indices.

> Code used:
```
df = df.sort_values(by='sugarpercent', ascending=False).reset_index(drop=True)
```

**Reflections and adds on**

We can go beyond simple sorting to explore multi-level sorting, sorting based on index, and custom sorting using lambda functions. Consider how sorting interacts with other DataFrame operations like grouping and merging, and its impact on data visualization.

Here's the general code:
- Sort by a single column: df.sort_values(by='column', ascending=True).
- Multi-level sorting: df.sort_values(by=['column1', 'column2'], ascending=[True, False]).



### Q9: Groupby + Agg (1.25 points) 
This section introduced grouping data and aggregating statistics within groups.

> Code used:
```
df.drop(columns='competitorname').groupby('chocolate').agg('mean')
sugar_summary = df.groupby('fruity')['sugarpercent'].agg(['min', 'mean', 'max'])
```

**Reflections and adds on:**

We can Investigate more complex groupby operations, such as grouping by multiple columns, using custom aggregation functions, and exploring pivot tables for cross-tabulation. Reflect on how these techniques enable deeper data exploration and hypothesis testing in datasets.

Here's the general code (I have provided an extra specific example for curiosity):
- Basic groupby: df.groupby('group_column').agg({'agg_column': 'sum'}).
- Complex aggregation: df.groupby(['column1', 'column2']).agg({'agg_column1': 'mean', 'agg_column2': 'max'}).
	In our case: df.groupby(['chocolate', 'fruity']).agg({'pricepercent': 'mean', 'winpercent': 'max'})


# A3
```
Quick intro: this assignment's purpose is to guide us through a complete data analysis from scratch, 
according to the guidance, I will break down this assignment into three parts, and talk about them separately.
```

### Part 1: Load & Clean Data (2.5 points) 
This part is mainly about the techniques for loading data into a DataFrame, identifying and addressing missing data, and cleaning messy data.

> .Apply( )
The first technique that worth memorizing would be the **apply** method, this method is very powerful when we want to transform the specific columns, we can self define some operation in the form of function, in our case, we define the "standardization" function, and apply it to the numerical features.

Code Used (a part):
```
df['year'] = df['year'].apply(standardize_year)
df['weight'] = df['weight'].apply(standardize_weight)
```
**Reflections and adds on:**

I'm particularly want to compare apply functions to "transform", and "filter" functions here:

- transform():
	- Purpose: Used when you want to transform data while keeping the original index. It's commonly used in group operations.
	- Behavior: The function passed to transform must return a result that is either the same size as the group chunk or broadcastable to the size of the group chunk (e.g., a scalar). transform is restricted compared to apply in terms of the kind of function you can use.
	- Use Cases: Particularly useful in groupby operations when you need to apply a function to each group and maintain the shape of the DataFrame.

- filter():
	- Purpose: Used to filter the data based on a specific criterion. This is different from boolean indexing: filter is used for selecting columns or rows based on their label.
	- Behavior: You can use it to filter for rows or columns based on their labels or index.
	- Use Cases: Useful when you want to select a subset of rows or columns based on their labels, not their content.


> Standardization (an other key takeaway from this assignment)

Standardization in the context of data processing refers to the practice of bringing different types or formats of data into a common framework. This is crucial in data analysis and machine learning for several reasons:

- Consistent Data Format: Standardization ensures that data from different sources or with different formats are transformed into a consistent format. This consistency is critical for effective data analysis and processing. For example, standardizing date formats (DD/MM/YYYY vs. MM/DD/YYYY), string formats (lowercase vs. uppercase), or measurement units (meters vs. feet).

- Feature Scaling: In machine learning, standardization often refers to scaling numerical values so that they have a mean of 0 and a standard deviation of 1. This is done using the formula: (original value - mean)/standard deviation. This is particularly important for models that are sensitive to the scale of input data, such as Support Vector Machines (SVMs) and k-Nearest Neighbors (k-NN).

- Handling Skewed Data: Standardization can also refer to transforming data to reduce skewness. Techniques like log transformations, square root transformations, or Box-Cox transformations can be used to make the data more "normal" (Gaussian), which is a common assumption in many statistical models.

- Improved Model Performance: Standardized data often leads to better performance in machine learning models. It ensures that features contribute equally to the model's prediction and prevents features with larger scales from dominating the learning algorithm's attention.

- Easier to Understand: Standardized data, especially in the context of z-scores, makes it easier to understand and interpret data points in terms of their distance from the mean in units of standard deviation.

**In summary, standardization is a critical step in data preprocessing that helps in making the data more uniform and suitable for analysis or machine learning models.**


> dropna(), and its parameter (subset = [])

The dropna() function in pandas is a very useful tool for handling missing data in a DataFrame or Series. When you have a dataset with missing values (NaN), dropna() allows you to remove rows and/or columns that contain missing data, which can be crucial for maintaining the integrity of your data analysis or machine learning models.

Basic code:
```
cleaned_df = df.dropna()
```

Code for drop columns:
```
cleaned_df = df.dropna(axis='columns')
```

The subset parameter of dropna() allows you to specify in which columns to look for missing values. This is particularly useful when you only want to drop rows based on missing values in specific columns rather than any column.

Code for using subset parameter:
```
cleaned_df = df.dropna(subset=['column1', 'column2'])
```

### Part 2: EDA - Exploratory Data Analysis (2.5 points) 
In this part, we have conducted the exploratory analysis, creating visualizations to understand data distributions and relationships, and interpreting these findings.

> Scatter Matrix plot

General code Used:
```
fig = pd.plotting.scatter_matrix(df)
```

Scatter matrix plot is a very useful tool, in which it can 
 - Visual Representation of Pairwise Relationships
 - be a base for Correlation Analysis
 	- Scatter plots in the matrix can reveal the nature of the relationship between variables, whether it’s linear, polynomial, or non-existent (no correlation).
	- The strength and direction of the relationship can also be inferred. Tight clusters of points indicate a strong relationship, while widely spread points suggest a weak or no relationship.
 - Identifying Outliers and Anomalies
 	- Outliers or unusual data points can be easily spotted in scatter plots. These are points that do not fit well with the general trend of the data.

 **In short, It can guide further data analysis steps, such as indicating which variables might require transformation, which pairs of variables might be useful for feature engineering, or which variables might need to be dropped due to redundancy.**


> Bar chart

Code used:
```
counts = df['major'].value_counts()
counts.plot(kind='bar', title='Number of Students in Each Major')
```

- Purpose: A bar plot is used to display and compare the frequency, count, or other measure (like mean) for different discrete categories or groups.
- Data Type: Bar plots are best for categorical data (like countries, brands, categories) or discrete data (like counts). (number of student in each major in our case)


> Histogram plot

Code used:
```
count = df[df['major'] == 'COGSCI']
plt.hist(count['height'])
```

- Purpose: A histogram can be used to show the distribution of a continuous dataset. It tells you how many data points fall into specific ranges (or "bins") of values.
- Data Type: Histograms are best for continuous data (like heights, weights, or ages). (height in our case)


### Part 3: Regression Analysis (3 points) 
This part of assignment is most valuable, guiding us to apply regression analysis to explore relationships between variables, interpret regression results, and understand the significance of these findings in the context of the research question.

> Linear Model Prediction of Height from Major

Code used:
```
formula = 'height ~ major'
outcome_1, predictors_1 = patsy.dmatrices(formula, df2)
mod_1 = sm.OLS(outcome_1, predictors_1)
res_1 = mod_1.fit()
```

The most important idea to take away from this part, I think would be the usage of dmtrices. In COGS 108 is the first time I used this concept, so I browsed through online to figure out its meanings under the context of linear regression: 

**It is important as many of our features (predictors) might be categorical Variables, which need to be converted into a form that can be used in the regression analysis, typically through a process called one-hot encoding or dummy coding.** dmatrices automatically handles this conversion. For example, if major has categories like 'Math', 'Science', and 'Engineering', dmatrices will create separate columns for each of these categories with binary values (0 or 1).

> Multivariate Regression for Height Prediction from Major and Gender

Formula used:
```
formula = 'height ~ major + gender'
```

the formula is the most tricky part when we go from single variable to multivaraite regression, so I provided the reference here.

> Interpret the result

How to interpret the result of regression analysis is also an important takeway from this assignment, how can we access and interpret the result of the OLS regression summary

Reference code:
```
result.summary() # where the result is the model after fit the data
```

After we get the summary table, to see whether the feature we choose have prediction power and significance for the dependent variable, we should look at the t value of the feature, compare it with the threshold we set. If its absolute value is strictly less than the threshold value, the it is significant. Besides that, how good the features we chose can explain the pattern can be seen from the R square (how much percentage of variance is explained)


### END

**This would be the END of this website, again, please do remember to triple check the submission after validation, do not leave a regret for being careless. That feels 1000% worse than losing the points for something you do not know. Take care.**
