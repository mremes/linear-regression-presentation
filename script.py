import numpy as np
import pandas as pd
import functools as ft
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, r2_score

# Fetch and shuffle data
datafile = "data.csv"
data = pd.read_csv(datafile)
data = shuffle(data)

# Clean data by removing null rows
filtered_data = data[~np.isnan(data["y"])]
x_y = np.array(filtered_data)

# Split data into x (predictors) and y (predicted value) and transpose
X, y = x_y[:, 0].reshape(-1, 1), x_y[:, 1].reshape(-1, 1)

# Split data into train and test
mask = np.random.rand(len(X)) < 0.8
X_train, X_test = X[mask], X[~mask]
y_train, y_test = y[mask], y[~mask]

# Pick model that can fit the data, for example linear regression: y = a*x + b
model = LinearRegression()
model.fit(X_train, y_train)

# Let's predict values for the test data and see how it performs
y_pred = model.predict(X_test)

# Let's create some util functions (showcasing FP)
fn = lambda to, val: round(val, to)
round2 = ft.partial(fn, 2)
round3 = ft.partial(fn, 3)

# Get some stats
a = model.coef_[0][0]
b = model.intercept_[0]
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

if r2 <= 0.5:
    result = 'poorly'
elif r2 <= 0.75:
    result = 'pretty well'
else:
    result = 'very well'

print(f"""
Function of this model is
    y = {round2(a)}x + {round2(b)}

Mean Squared Error is
    {round3(mse)}

Variance score R2 is
    {round2(r2)}, which means x explains y {result}.
""")


# Finally, let's plot the results
plt.scatter(X_train, y_train, color='blue')
plt.plot(X_test, y_pred, color='red', linewidth=3)
plt.show()
