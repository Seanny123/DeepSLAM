# Create the bar chart for the averaging comparison
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(font_scale=1.5)

layer_names = ['GoogleNet Layer 1', 'AlexNet Layer 2',
               'OverFeat Average', 'All Layers',
               'Best Combination']

results = pd.DataFrame()
results = results.append({
                          'Layer': layer_names[0],
                          'F1 Score': 0.85260511883,
                          'Boosted': True,
                         }, ignore_index=True)
results = results.append({
                          'Layer': layer_names[1],
                          'F1 Score': 0.841242350768,
                          'Boosted': True,
                         }, ignore_index=True)
results = results.append({
                          'Layer': layer_names[2],
                          'F1 Score': 0.783751493429,
                          'Boosted': True,
                         }, ignore_index=True)
results = results.append({
                          'Layer': layer_names[3],
                          'F1 Score': 0.870800450958,
                          'Boosted': True,
                         }, ignore_index=True)
results = results.append({
                          'Layer': layer_names[4],
                          'F1 Score': 0.882036331016,
                          'Boosted': True,
                         }, ignore_index=True)


bar = sns.factorplot('Layer', 'F1 Score', data=results,
                     kind='bar', size=8, legend=False,
                     order=layer_names)
axes = bar.axes[0, 0]
axes.set_title('Combined Improvement')
axes.set_ylim(0.7, 0.9)

plt.show()
