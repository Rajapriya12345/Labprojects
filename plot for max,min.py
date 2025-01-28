import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('employees.csv')


data = df["Salary"].values


average_value = data.mean()
max_value = data.max()
min_value = data.min()


df_stats = pd.DataFrame({
    'Stat': ['Average', 'Max', 'Min'],
    'Value': [average_value, max_value, min_value]
})


df_stats.plot(kind='bar', x='Stat', y='Value', figsize=(10, 5))


plt.xlabel('Statistics')
plt.ylabel('Salary')
plt.title('Average, Max, and Min Salary Values')


plt.tight_layout()
plt.savefig('average_max_min_plot.png')  


plt.show()
