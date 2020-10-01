import numpy as np
import matplotlib.pyplot as plt
import csv

plt.rcParams.update({'figure.figsize': [6.5, 5.25]})
plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'font.size': 14})

plt.figure()
plt.grid(True)
plt.xlabel('Runtime per 1000 coarse-grid (32 x 32) steps (in seconds)')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('Accuracy-speed tradeoff')
for resolution in [64, 128]:
  with open(f'stats_{resolution}.txt') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
      markersize = 14
      color = 'red'
      if resolution == 64:
        color = 'green'
      plt.semilogx(float(row[0]), float(row[1]), color=color, marker='o', markersize=markersize, label=f'Baseline {resolution} x {resolution}')
for stencil_size in [3, 5]:
  for num_layers in [4, 6, 8]:
    for filters in [64, 128]:
      with open(f'stats_{stencil_size}-{num_layers}-{filters}.txt') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
          label = ''
          if stencil_size == 3 and num_layers == 4 and filters == 64:
            label = 'Neural net 32 x 32\n(different hyperparameters)'
          plt.semilogx(float(row[0]), float(row[1]), 'b*', label=label)
plt.legend()
plt.savefig('mae_runtime.png')
