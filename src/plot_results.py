import pandas as pd
from datetime import datetime
import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import seaborn as sns

output_name = "output"


def plot_data(df, x_label, y_label, graph_title):
    for method in uniq_methods:
        
        df2 = df[ df['Method'] == method ]
        
        x = df2[x_label]
        y = df2[y_label]
     
        # plot
        plt.plot(x,y, label=method)
        

    plt.gcf().autofmt_xdate()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc="upper left")
    plt.title(graph_title)
    plt.show()
    
    
def histo_plot(df, x_label, y_label, graph_title):
    
   qual = df.groupby("Method").agg([np.mean, np.std])
   qual = qual[x_label]
   qual.plot(kind = "barh", y = "mean", legend = False, 
          xerr = "std", title = graph_title, color='blue')
   plt.xlabel(x_label)
   plt.ylabel(y_label)
   plt.show()
   
def histo_plot_count(df, x_label, y_label, graph_title):
    
   
   df['percent'] = df[x_label]/len(df.index) * len(uniq_methods) * 100
   
    
   qual = df.groupby("Method").agg({"percent": [ 'sum']})
   
   qual = qual['percent']
  
   
   qual.plot(kind = "barh", legend = False, 
          title = graph_title, color='green')
   plt.xlabel(f'Percent of {x_label}')
   plt.ylabel(y_label)
   plt.show()
   
   
   
template_sizes = [128, 256]
altered_types = ['normal','blur', 'zoom', 'rotation', 'noise']

for t_size in template_sizes:   
    for a_type in altered_types:
        df = pd.read_csv(f"../results/{output_name}_{a_type}_{t_size}.csv",header=0, dtype=object,sep=';')
        
        #on supprime le valeurs nulles
        df = df[df['Execution time (s)'] != 'None']

        df['Execution time (s)'] = df['Execution time (s)'].astype(float)
        df['Valid match'] = df['Valid match'].map({'True': True, 'False': False})
        #df['Valid match'] = df['Valid match'].astype(int)
        
        #List unique values in the df['Method'] column
        uniq_methods = df['Method'].unique()
          
        plot_data(df, 'Image number', 'Execution time (s)',f"Execution time comparison ({a_type} {t_size}*{t_size} )" )
        histo_plot(df, 'Execution time (s)', 'Methods',f'Execution time comparison ({a_type} {t_size}*{t_size})' )
        histo_plot_count(df, 'Valid match', 'Methods',f'Valid match comparison ({a_type} {t_size}*{t_size})' )
   