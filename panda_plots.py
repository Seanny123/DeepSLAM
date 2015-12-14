
import numpy as np
import ipdb
import sys
import cPickle as pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# dicts for converting layer name to num
vgg19_num = {'conv4_4':1, 'conv5_4':2}

googlenet_num = {'3a':1,'3b':2,'4a':3,'4b':4,'4c':5,'4d':6,'4e':7,'5a':8,'5b':9}


fname = 'filter_res_avg.p'
fname = 'filter_res_all.p'

data = pickle.load(open(fname, 'rb'))

results = pd.DataFrame()

for name in data.keys():
  if 'googlenet' in name:
    net_type = 'GoogLeNet'
    layer_name = name[40:42]
    layer_num = googlenet_num[layer_name]
  elif 'overfeat' in name:
    net_type = 'OverFeat'
    layer_name = name[29:-3]
    layer_num = int(layer_name)
  elif 'vgg19' in name:
    net_type = 'VGG19'
    layer_name = name[26:-3]
    layer_num = vgg19_num[layer_name]
  elif 'caffenet' in name:
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
                            'Net': net_type,
                            'Layer': layer_name,
                            'Layer #': layer_num,
                            'Precision': data[name][0],
                            'Recall': data[name][1],
                            'F1': data[name][2],
                            'Boosted': False,
                           }, ignore_index=True)
  
  results = results.append({'Name': name, # TODO: get rid of conf_mat crap
                            'Net': net_type,
                            'Layer': layer_name,
                            'Layer #': layer_num,
                            'Precision': data[name][3],
                            'Recall': data[name][4],
                            'F1': data[name][5],
                            'Boosted': True,
                           }, ignore_index=True)

googlenet = results[(results['Net'] == 'GoogLeNet')]
overfeat = results[(results['Net'] == 'OverFeat')]
cifar10 = results[(results['Net'] == 'Cifar10')]
alexnet = results[(results['Net'] == 'AlexNet')]
vgg19 = results[(results['Net'] == 'VGG19')]

bar = sns.factorplot('Layer #', 'F1', 'Boosted', data=googlenet, kind='bar', size=6,
                     legend=True)
bar = sns.factorplot('Layer #', 'F1', 'Boosted', data=overfeat, kind='bar', size=6,
                     legend=True)
bar = sns.factorplot('Layer #', 'F1', 'Boosted', data=cifar10, kind='bar', size=6,
                     legend=True)
bar = sns.factorplot('Layer #', 'F1', 'Boosted', data=alexnet, kind='bar', size=6,
                     legend=True)
bar = sns.factorplot('Layer #', 'F1', 'Boosted', data=vgg19, kind='bar', size=6,
                     legend=True)


bar = sns.factorplot('Layer', 'F1', 'Boosted', data=results, kind='bar', size=6,
                     legend=True)

plt.show()
