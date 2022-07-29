import numpy as np


f = open('C:/Users/Chaputa/Documents/Trier/Master/Masterarbeit/Daten_Polar/test/Polar_Sense_B2324727_20220719_100518_PPG.txt', 'r')
lines = f.readlines()
lines.pop(0)

Channel0 = []
Channel1 = []
Channel2 = []
Ambient = []
for x in lines:
    Channel0.append(x.split(';')[2])
    Channel1.append(x.split(';')[3])
    Channel2.append(x.split(';')[4])
    Ambient.append(x.split(';')[5])
f.close()
# converting list to array
Channel0 = np.array(Channel0)
Channel1 = np.array(Channel1)
Channel2 = np.array(Channel2)
Ambient = np.array(Ambient)
Channel0 = Channel0.astype(np.int32)
Channel1 = Channel1.astype(np.int32)
Channel2 = Channel2.astype(np.int32)
Ambient = Ambient.astype(np.int32)

# (Channel0 Channel1 Channel2 - (3 * Ambient))/3
PPG = (Channel0 + Channel1 + Channel2 - (3 * Ambient))/3
