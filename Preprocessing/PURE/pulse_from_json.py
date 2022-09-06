import json
import os
import numpy as np

# Opening JSON file save all in data as dict
path = r"C:\Users\Chaputa\Documents\Trier\Master\Masterarbeit\rPPGCode\Preprocessing\PURE\10-06.json"
f = open(path)
data = json.load(f)
f.close()

#%% save all pulse values ('waveform') in bvp array
bvp = np.zeros(len(data['/FullPackage']))
idx = 0
# if '/FullPackage'
#   loop over data elements ->
#       if 'Value'
#           if 'waveform'
#               save in bvp array
for FullPackage_Image, values1 in data.items():
    if FullPackage_Image == '/FullPackage':
        for values2 in values1:
            for key3, values3 in values2.items():
                if key3 == "Value":
                    for key4, values4 in values3.items():
                        if key4 == "waveform":
                            bvp[idx] = values4
                            idx = idx + 1


