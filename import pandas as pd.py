import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Load the data from the Excel file
file_path = r'C:\Users\McHenry Nchekwaram\Downloads\baseball.xlsx'
data = pd.read_excel(file_path)

# First linear regression: Wins vs Runs Difference (Runs Scored - Runs Allowed)
x1 = data['Runs Scored'] - data['Runs Allowed'] # Independent variable
y1 = data['Wins'] # Dependent variable

# Perform linear regression
slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(x1, y1)

# Create regression line
regression_line1 = slope1 * x1 + intercept1

# Visualization
plt.figure(figsize=(8, 6))
plt.scatter(x1, y1, color='blue', label='Data Points')
plt.plot(x1, regression_line1, color='red', label='Regression Line')
plt.xlabel('Runs Difference (Runs Scored - Runs Allowed)')
plt.ylabel('Wins')
plt.title('Wins vs Runs Difference Regression')
plt.text(0.05, 0.95, f'R-squared = {r_value1**2:.2f}', transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top')
plt.legend()
plt.grid(True)
plt.show()

# Second linear regression: Runs Difference (Runs Scored - Runs Allowed) vs Team Batting Average
x2 = data['Team Batting Average'] # Independent variable
y2 = data['Runs Scored'] - data['Runs Allowed'] # Dependent variable

# Perform linear regression
slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(x2, y2)

# Create regression line
regression_line2 = slope2 * x2 + intercept2

# Visualization
plt.figure(figsize=(8, 6))
plt.scatter(x2, y2, color='blue', label='Data Points')
plt.plot(x2, regression_line2, color='red', label='Regression Line')
plt.xlabel('Team Batting Average')
plt.ylabel('Runs Difference (Runs Scored - Runs Allowed)')
plt.title('Runs Difference vs Team Batting Average Regression')
plt.text(0.05, 0.95, f'R-squared = {r_value2**2:.2f}', transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top')
plt.legend()
plt.grid(True)
plt.show()

# Third multiple regression: Runs Difference vs OBP and SLG
x3 = data[['OBP', 'SLG']] # Independent variables
y3 = data['Runs Scored'] - data['Runs Allowed'] # Dependent variable

# Add constant for multiple regression
x3 = np.column_stack((np.ones(len(x3)), x3))

# Perform multiple regression
coefficients = np.linalg.lstsq(x3, y3, rcond=None)[0]
predicted_y3 = np.dot(x3, coefficients)

# Print regression statistics
print('Multiple Regression Statistics:')
print(f'Intercept: {coefficients[0]}')
print(f'Coefficient for OBP: {coefficients[1]}')
print(f'Coefficient for SLG: {coefficients[2]}')

# Predict if a team would make it into the playoffs based on RS, RA, W, OBP, SLG, and BA
# This could be done using logistic regression or classification algorithms

