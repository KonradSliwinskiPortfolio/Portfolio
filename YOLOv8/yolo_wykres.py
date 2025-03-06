import os
import pandas as pd
import matplotlib.pyplot as plt


results_path = './runs/classify/train6/results.csv'

results = pd.read_csv(results_path)

plt.figure()
plt.plot(results['                  epoch'], results['             train/loss'], label='train loss')
plt.plot(results['                  epoch'], results['               val/loss'], label='val loss', c='red')
plt.grid()
plt.title('strata podczas treningu')
plt.ylabel('strata')
plt.xlabel('epoki')
plt.legend()
plt.savefig('50_2_wykres1.pdf')
plt.show()


plt.figure()
plt.plot(results['                  epoch'], results['  metrics/accuracy_top1'] * 100)
plt.grid()
plt.title('dokladnosc walidacji w trakcie procesu szkolenia')
plt.ylabel('dokladnosc (%)')
plt.xlabel('epoki')


plt.savefig('50_2_wykres2.pdf')
plt.show()