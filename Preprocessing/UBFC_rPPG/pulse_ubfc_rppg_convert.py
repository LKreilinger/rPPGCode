import numpy as np
import pandas

def pulse_convert(current_path, tempPathNofile):
    f = open(current_path, 'r')
    lines = f.readlines()
    lines.pop(1)
    lines.pop(1)

    f.close()
    string_label = lines[0]
    bvp_label = string_label.split()
    #%% save bvp_label as *.csv
    newDF = pandas.DataFrame(bvp_label)
    newDF.to_csv(tempPathNofile, index=False, header=False)
    #todo save as csv 1 coloum