import pandas as pd
from datetime import datetime
import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

df = pd.read_csv('../results/output_blur_256.csv',header=0, dtype=object,sep=';')
#print (df)

df['Execution time (s)'] = df['Execution time (s)'].astype(float)
 

#List unique values in the df['Method'] column
uniq_methods = df['Method'].unique()

data_per_method = {}


def plot_data(df, x_label, y_label, graph_title):
    for method in uniq_methods:
        
        df2 = df[ df['Method'] == method ]
        
        x = df2[x_label]
        y = df2[y_label]
    
        data_per_method[method] = (x,y)    
        # plot
        plt.plot(x,y, label=method)

    plt.gcf().autofmt_xdate()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc="upper left")
    plt.title(graph_title)
    plt.show()

    #print(data_per_method)

plot_data(df, 'Image number', 'Execution time (s)',"Execution time comparison (Blur 256*256px template )" )