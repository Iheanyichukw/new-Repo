import wbdata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
from datetime import datetime
from sklearn.linear_model import LinearRegression

"""
    The data in this script is fetched directly from the 
    World Bank's database via the wbdata.get_dataframe() function, this 
    is to authomatically set up the data so that all informations will be 
    avaliable during the analysis"""
    
# Set the indicators to retrieve
indicators = {
    "EN.ATM.CO2E.PC": "CO2_per_capita",
    "NY.GDP.PCAP.CD": "GDP_per_capita"
}

# Define the date range for the data
start_date = datetime(2000, 1, 1)
end_date = datetime(2020, 12, 31)

# Fetch the data
data = wbdata.get_dataframe(indicators, data_date=(
    start_date, end_date), convert_date=False)

# Reset the index and rename the columns
data = data.reset_index().rename(
    columns={"country": "Country", "date": "Year"})

# Merge dataframes on the 'Country' and 'Year' columns
df_co2 = data[['Country', 'Year', 'CO2_per_capita']]
df_gdp = data[['Country', 'Year', 'GDP_per_capita']]

df = df_co2.merge(df_gdp, on=['Country', 'Year'])

# Fill missing data with the mean of the respective column
df['CO2_per_capita'] = df['CO2_per_capita'].fillna(df['CO2_per_capita'].mean())
df['GDP_per_capita'] = df['GDP_per_capita'].fillna(df['GDP_per_capita'].mean())

# Normalize the data for clustering
scaler = StandardScaler()
normalized_data = scaler.fit_transform(
    df[['CO2_per_capita', 'GDP_per_capita']])

"""
    the statitical analysis was employed to create a pathway for analysis
    and the statistical standing of the dataset we used"""
    

# Generate summary statistics using the .describe() method
summary = df.describe()

# Print or view the summary DataFrame
print(summary)

# Normalize the data for clustering
scaler = StandardScaler()
normalized_data = scaler.fit_transform(
    df[['CO2_per_capita', 'GDP_per_capita']])

"""
    Cluster analysis or clustering is the task of grouping a set of objects 
    in such a way that objects in the same group (called a cluster) 
    are more similar (in some sense) to each other than to those in other 
    groups (clusters).
    we shall be using Kmeans in carrying out clustering analysis.
    the aim is the find out the overall correlation between CO2 per capital 
    and GDP per capital"""
    
# Perform clustering using KMeans
num_clusters = 4  # Choose the number of clusters
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(normalized_data)

# Assign cluster labels to each data point
df['Cluster'] = kmeans.labels_

# Visualize the clusters using a scatter plot
plt.scatter(df['GDP_per_capita'], df['CO2_per_capita'],
            c=df['Cluster'], cmap='viridis')
plt.xlabel('GDP per capita')
plt.ylabel('CO2 per capita')
plt.title('Clusters of Countries by GDP and CO2 Emissions')
plt.show()

# Calculate cluster centers
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)

# Add cluster centers to the scatter plot
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
            c='red', marker='x', s=100, label='Cluster Centers')

# Update plot settings
plt.xlabel("GDP per capita")
plt.ylabel("CO2 per capita")
plt.title("Clusters of Countries by GDP and CO2 Emissions")
plt.legend()
plt.show()

# Apply KMeans clustering algorithm
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(normalized_data)

# Add cluster labels to the original dataframe
df['Cluster'] = kmeans.labels_

# Visualize the clusters
plt.scatter(df['GDP_per_capita'], df['CO2_per_capita'],
            c=df['Cluster'], cmap='viridis')
plt.xlabel("GDP per capita")
plt.ylabel("CO2 per capita")
plt.title("CO2 per capita vs GDP per capita")
plt.show()

for cluster in df['Cluster'].unique():
    print(f"Cluster {cluster}:")
    print(df[df['Cluster'] == cluster]
          [['Country', 'CO2_per_capita', 'GDP_per_capita']])
    print("\n")

cluster_summary = df.groupby(
    'Cluster')[['CO2_per_capita', 'GDP_per_capita']].mean()
print(cluster_summary)

# List the countries in each cluster
for i in range(df['Cluster'].nunique()):
    cluster_countries = df[df['Cluster'] == i]['Country']
    print(f"Cluster {i}:")
    print(cluster_countries)
    print()


def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers


# Identify outliers for CO2 emissions per capita and GDP per capita
outliers_co2 = detect_outliers(df, 'CO2_per_capita')
outliers_gdp = detect_outliers(df, 'GDP_per_capita')

# Merge the outlier dataframes
outliers = pd.concat([outliers_co2, outliers_gdp]).drop_duplicates()

# Print the outlier countries
print("Outlier countries:")
print(outliers['Country'])

correlation = df['CO2_per_capita'].corr(df['GDP_per_capita'])
print("Correlation between CO2 per capita and GDP per capita:", correlation)

# Compute correlation coefficients within each cluster
for cluster_label in df['Cluster'].unique():
    cluster_df = df[df['Cluster'] == cluster_label]
    correlation = cluster_df['CO2_per_capita'].corr(
        cluster_df['GDP_per_capita'])
    print(
        f"Correlation between CO2 per capita and GDP per capita in Cluster {cluster_label}: {correlation}")


# Fit a linear regression model
X = df[['GDP_per_capita']]
y = df['CO2_per_capita']
reg = LinearRegression().fit(X, y)

# Predict CO2 per capita using the linear regression model
df['Predicted_CO2_per_capita'] = reg.predict(X)

# Visualize the linear regression model's predictions
plt.scatter(df['GDP_per_capita'], df['CO2_per_capita'],
            c=df['Cluster'], cmap='viridis', label='Actual')
plt.scatter(df['GDP_per_capita'], df['Predicted_CO2_per_capita'],
            c='red', marker='x', label='Predicted')
plt.xlabel("GDP per capita")
plt.ylabel("CO2 per capita")
plt.title("CO2 per capita vs GDP per capita (Linear Regression)")
plt.legend()
plt.show()

# Fit linear regression models for each cluster
models = {}
for cluster in df['Cluster'].unique():
    cluster_data = df[df['Cluster'] == cluster]
    X = cluster_data['GDP_per_capita'].values.reshape(-1, 1)
    y = cluster_data['CO2_per_capita'].values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(X, y)
    models[cluster] = model

# Add the predicted CO2 per capita values to the dataframe
df['Predicted_CO2_per_capita'] = np.nan
for cluster, model in models.items():
    cluster_indices = df[df['Cluster'] == cluster].index
    X = df.loc[cluster_indices, 'GDP_per_capita'].values.reshape(-1, 1)
    predictions = model.predict(X)
    df.loc[cluster_indices,
           'Predicted_CO2_per_capita'] = predictions.reshape(-1)

gdp_value = 40000
predictions = {}

for cluster, model in models.items():
    prediction = model.predict([[gdp_value]])
    predictions[cluster] = prediction[0][0]

print(predictions)

# Visualize the linear regression model's predictions
plt.scatter(df['GDP_per_capita'], df['CO2_per_capita'],
            c=df['Cluster'], cmap='viridis', label='Actual')
plt.scatter(df['GDP_per_capita'], df['Predicted_CO2_per_capita'],
            c='red', marker='x', label='Predicted')
plt.xlabel("GDP per capita")
plt.ylabel("CO2 per capita")
plt.title("CO2 per capita vs GDP per capita (Linear Regression)")
plt.legend()
plt.show()


# ============================
# Simple Model Fitting Section
# ============================


def model_func(x, a, b):
    return a * x + b

"""
 Calculates the upper and lower limits for the function, parameters and
 sigmas for single value or array x. Functions values are calculated for 
 all combinations of +/- sigma and the minimum and maximum is determined.
 Can be used for all number of parameters and sigmas >=1.

 This routine can be used in assignment programs.
 """
def err_ranges(popt, pcov, x):
    a, b = popt
    sigma_a, sigma_b = np.sqrt(np.diag(pcov))
    y_fit = model_func(x, a, b)
    delta_y = np.sqrt((x * sigma_a) ** 2 + sigma_b ** 2)
    return y_fit - delta_y, y_fit + delta_y


# Using df from the previous code
df['Year'] = df['Year'].apply(lambda x: datetime.strptime(x, "%Y"))

# Prepare data for curve fitting
x_data = df['Year'].apply(lambda x: x.year - 2000).values
y_data = df['CO2_per_capita'].values

# Fit the curve
popt, pcov = curve_fit(model_func, x_data, y_data)

# Predictions for 10 and 20 years into the future
x_future = np.array([30, 40])
y_future = model_func(x_future, *popt)

# Confidence intervals
lower, upper = err_ranges(popt, pcov, x_data)

# Visualize the data, fitted curve, and confidence intervals
plt.scatter(x_data, y_data, label='Data', alpha=0.5)
plt.plot(x_data, model_func(x_data, *popt), 'r-', label='Fitted Curve')
plt.fill_between(x_data, lower, upper, color='gray',
                 alpha=0.5, label='Confidence Range')

plt.xlabel("Years since 2000")
plt.ylabel("CO2 per capita")
plt.title("CO2 per capita vs Years (Simple Linear Model)")
plt.legend()
plt.show()

print(f"Predicted CO2 per capita in 10 years (2033): {y_future[0]:.2f}")
print(f"Predicted CO2 per capita in 20 years (2043): {y_future[1]:.2f}")

# ============================
# Comparison of trends in different clusters
# ============================


def plot_trend(cluster, countries, feature):
    plt.figure(figsize=(10, 5))
    for country in countries:
        country_data = df[df['Country'] == country]
        plt.plot(country_data['Year'], country_data[feature], label=country)

    plt.xlabel("Year")
    plt.ylabel(feature)
    plt.title(f"Trends for {feature} in Cluster {cluster}")
    plt.legend()
    plt.show()


# Select one country from each cluster
country_per_cluster = df.groupby('Cluster')['Country'].first().tolist()

# Plot trends for CO2 per capita in each cluster
for i, country in enumerate(country_per_cluster):
    plot_trend(i, [country], 'CO2_per_capita')

# Choose two countries within the same cluster (e.g., Cluster 0)
countries_same_cluster = df[df['Cluster'] == 0]['Country'].head(2).tolist()

# Plot trends for CO2 per capita for the two countries within the same cluster
plot_trend(0, countries_same_cluster, 'CO2_per_capita')

# Choose one country from Cluster 0 and one country from Cluster 1 for comparison
countries_diff_cluster = [df[df['Cluster'] == 0]
                          ['Country'].iloc[0], df[df['Cluster'] == 1]['Country'].iloc[0]]


# Extract representative countries from each cluster
representative_countries = {}
for i in range(df['Cluster'].nunique()):
    cluster_countries = df[df['Cluster'] == i][[
        'Country', 'CO2_per_capita', 'GDP_per_capita']]
    # Pick one country from each cluster, e.g., the country with the median CO2 emissions
    representative_country = cluster_countries.loc[cluster_countries['CO2_per_capita'].idxmin(
    )]
    representative_countries[i] = representative_country

# Print representative countries and their key statistics
for cluster, country_info in representative_countries.items():
    print(
        f"Cluster {cluster}: {country_info['Country']} (CO2 per capita: {country_info['CO2_per_capita']:.2f}, GDP per capita: {country_info['GDP_per_capita']:.2f})")
