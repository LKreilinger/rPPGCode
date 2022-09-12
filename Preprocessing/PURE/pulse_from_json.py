import json
import os
import numpy as np
import pandas

def pulse_pure(currentPath, destination_path):
    # Opening JSON file save all in data as dict
    f = open(currentPath)
    data = json.load(f)
    f.close()

    #%% save all pulse values ('waveform') in bvp array and timestamps
    bvp = np.zeros(len(data['/FullPackage']))
    timestamp = np.zeros(len(data['/FullPackage']))
    idx = 0
    # if '/FullPackage'
    #   loop over data elements ->
    #       if 'Value'
    #           if 'waveform'
    #               save bvp
    for FullPackage_Image, values1 in data.items():
        if FullPackage_Image == '/FullPackage':
            for values2 in values1:
                for key3, values3 in values2.items():
                    if key3 == "Value":
                        for key4, values4 in values3.items():
                            if key4 == "waveform":
                                bvp[idx] = values4
                                idx = idx + 1
    #%% save bvp data as *.csv
    newDF = pandas.DataFrame(bvp)
    newDF.to_csv(destination_path, index=False, header=False)


