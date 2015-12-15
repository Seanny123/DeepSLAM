
import numpy as np
import ipdb
import sys
import cPickle as pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(font_scale=1.5)

# dicts for converting layer name to num
vgg19_num = {'conv4_4':1, 
             'conv5_3':2, 'conv5_4':3}

googlenet_num = {'3a':1,'3b':2,'4a':3,'4b':4,'4c':5,'4d':6,'4e':7,'5a':8,'5b':9}


fname = 'filter_res_all.p'

data = pickle.load(open(fname, 'rb'))

results = pd.DataFrame()

for name in data.keys():
  if 'googlenet' in name:
    net_type = 'GoogLeNet'
    layer_name = name[40:42]
    layer_num = layer_name#googlenet_num[layer_name]
  elif 'overfeat' in name:
    net_type = 'OverFeat'
    layer_name = name[29:-3]
    layer_num = int(layer_name)
  elif 'vgg19' in name:
    net_type = 'VGG19'
    layer_name = name[26:-3]
    layer_num = vgg19_num[layer_name]
  elif ('caffenet' in name) or ('alexnet' in name):
    net_type = 'AlexNet'
    layer_name = name[29:-3]
    layer_num = int(layer_name[4:])
  elif 'cifar10full' in name:
    net_type = 'Cifar10'
    layer_name = 'cifar_' + name[32:-3]
    layer_num = int(layer_name[10:])
  elif 'average' in name:
    net_type = 'Average'
    layer_name = 'average'
    layer_num = 1
  else:
    net_type = 'NOTFOUND'
    layer_name = 'notfound'
  results = results.append({'Name': name, # TODO: get rid of conf_mat crap
                            'Network': net_type,
                            'Layer': layer_name,
                            'Layer #': layer_num,
                            'Precision': data[name][0],
                            'Recall': data[name][1],
                            'F1 Score': data[name][2],
                            'Boosted': False,
                            'AVG':False,
                           }, ignore_index=True)
  
  results = results.append({'Name': name, # TODO: get rid of conf_mat crap
                            'Network': net_type,
                            'Layer': layer_name,
                            'Layer #': layer_num,
                            'Precision': data[name][3],
                            'Recall': data[name][4],
                            'F1 Score': data[name][5],
                            'Boosted': True,
                            'AVG':False,
                           }, ignore_index=True)

# Get the average data
fname = 'filter_res_avg.p'
data = pickle.load(open(fname, 'rb'))
for name in data.keys():
  layer_name = None
  layer_num = None
  if 'googlenet' in name:
    net_type = 'GoogLeNet'
  elif 'overfeat' in name:
    net_type = 'OverFeat'
  elif 'vgg19' in name:
    net_type = 'VGG19'
  elif ('caffenet' in name) or ('alexnet' in name):
    net_type = 'AlexNet'
  elif 'cifar10' in name:
    net_type = 'Cifar10'
  elif 'all_net' in name:
    net_type = 'NetworkAverage'
    continue
  elif 'all_layer' in name:
    net_type = 'LayerAverage'
    net_type = 'Combined'
  else:
    net_type = 'NOTFOUND'
    layer_name = 'notfound'
    continue
  results = results.append({'Name': name, # TODO: get rid of conf_mat crap
                            'Network': net_type,
                            'Layer': layer_name,
                            'Layer #': layer_num,
                            'Precision': data[name][0],
                            'Recall': data[name][1],
                            'F1 Score': data[name][2],
                            'Boosted': False,
                            'AVG':True,
                           }, ignore_index=True)
  
  results = results.append({'Name': name, # TODO: get rid of conf_mat crap
                            'Network': net_type,
                            'Layer': layer_name,
                            'Layer #': layer_num,
                            'Precision': data[name][3],
                            'Recall': data[name][4],
                            'F1 Score': data[name][5],
                            'Boosted': True,
                            'AVG':True,
                           }, ignore_index=True)

if True:#fname == 'filter_res_avg.p':
  kind = 'bar'
  title = 'Layer Averages'
  order = ['Cifar10', 'AlexNet', 'OverFeat', 'GoogLeNet', 'VGG19', 'Combined']

  if True: # Show max and boosted only
    results = results[(results['Boosted'] == True)]

    for metric in ['F1 Score', 'Precision', 'Recall']:
      # Only look at non-averages for max
      max_val = results[(results['AVG'] == False)]
      max_val = max_val.groupby(['Network'], sort=False)['F1 Score'].max()
      
      max_add = pd.DataFrame()
      for name, value in max_val.iteritems():
        max_add = max_add.append({'Network':name,
                                  metric:value,
                                  'AVG':False,
                                 }, ignore_index=True)

      avg_results = results[(results['AVG'] == True)]

      avg_results = pd.concat([max_add, avg_results])

      bar = sns.factorplot('Network', metric, 'AVG', data=avg_results, kind=kind, size=6,
                           legend=False, order=order)
      
      bar.axes[0,0].set_title(title)
      handles, labels = bar.axes[0,0].get_legend_handles_labels()
      bar.axes[0,0].legend(handles, ['Max', 'Average'], loc='center left', bbox_to_anchor=(1, 0.5))

  else: # show boosted and non-boosted and not max
    results = results[(results['AVG'] == True)]
    bar = sns.factorplot('Network', 'F1 Score', 'Boosted', data=results, kind=kind, size=6,
                         legend=True, order=order)
    bar.axes[0,0].set_title(title)
    
    bar = sns.factorplot('Network', 'Precision', 'Boosted', data=results,
                         kind=kind, size=6, legend=True, order=order)
    bar.axes[0,0].set_title(title)
    
    bar = sns.factorplot('Network', 'Recall', 'Boosted', data=results, kind=kind, size=6,
                         legend=True, order=order)
    bar.axes[0,0].set_title(title)

else:
  results = results[(results['AVG'] == False)]
  kind = 'point'
  network_names = ['GoogLeNet', 'OverFeat', 'Cifar10', 'AlexNet', 'VGG19']
  for name in network_names:

    net = results[(results['Network'] == name)]
    bar = sns.factorplot('Layer #', 'F1 Score', 'Boosted', data=net,
                         kind=kind, size=8, legend=True)
    bar.axes[0,0].set_title(name)
    if name != 'GoogLeNet':
      labels = [int(item.get_text()) for item in bar.axes[0,0].get_xticklabels()]
      bar.axes[0,0].set_xticklabels(labels)

  """
  kind = 'bar'
  bar = sns.factorplot('Layer', 'F1 Score', 'Boosted', data=results, kind=kind, size=6,
                       legend=True)
  bar.axes[0,0].set_title('All Layers')
  """
plt.show()
