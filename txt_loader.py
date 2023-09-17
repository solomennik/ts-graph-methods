import glob
import os
import pandas as pd

def txt_loader(path):

    file_list = glob.glob(os.path.join(os.getcwd(), path, "*.txt"))

    df = pd.DataFrame()

    for txt_file in file_list:
        table = pd.read_csv(txt_file, delimiter = "\t")
        try:
            table = table[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
            table['Company'] = str(txt_file[49:-4])
            #print(str(txt_file[49:-4]))
            if len(table.columns) > 5:
                df = df.append(table, ignore_index=True)
        except KeyError:
            pass        
        
    return df
