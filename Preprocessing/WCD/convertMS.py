import datetime
ms = 1658833754646                                       # Example milliseconds object
time1 = datetime.datetime.fromtimestamp(ms / 1000)  # Apply fromtimestamp function
print(time1)

ms = 1658833935743                                       # Example milliseconds object
time2 = datetime.datetime.fromtimestamp(ms / 1000)  # Apply fromtimestamp function
print(time2)

diff = time2 - time1
print(diff)
#%%
import datetime
import time
# st = time.strptime("01.01.2012", "%d.%m.%Y")
# epoch = time.asctime(st)
# print("epoch is:", epoch)
dt = datetime.datetime.fromtimestamp(712155798298598141 // 1000000000)
s1 = dt.strftime('%Y-%m-%d %H:%M:%S')
s1 += '.' + str(int(1360287003083988472 % 1000000000)).zfill(9)
print(s1)
#datetime_object = datetime.strptime(s1, '%y %m %d %H:%M:%S')

ms = 712155876615892589                                       # Example milliseconds object
time2 = datetime.datetime.fromtimestamp(ms / 1000000000)  # Apply fromtimestamp function
print(time2)

dt = datetime.datetime.fromtimestamp(712155798298598141 // 1000000000)
s2 = dt.strftime('%Y-%m-%d %H:%M:%S')
s2 += '.' + str(int(1360287003083988472 % 1000000000)).zfill(9)
print(s2)

diff = s1 - s2
print(diff)
#%%
# Generate harder answer list
list_Number_sound = open("answers.txt", "r")
data = list_Number_sound.read()
list_Number_sound.close()
list_wavs = data.split('\n')
del list_wavs[-1]
answers_plus_one = [str(int(i) + 2) for i in list_wavs]

textfile = open("answers_plus_one.txt", "w")

for element in answers_plus_one:

    textfile.write(element + "\n")

textfile.close()
