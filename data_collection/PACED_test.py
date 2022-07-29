import time
import os
from playsound import playsound
import keyboard

print("Press 1 on keyboard to start playing audios: Level normal")
print("Press 2 on keyboard to start playing audios: Level hard")
print("Press c on keyboard to stop playing audios")
path_wav = "C:\\Users\\Chaputa\\Documents\\Trier\\Master\\Masterarbeit\\misc\\Versuch_audio_files\\"
#%% First 3min
# opening the list with the order in which the *.wav are played
list_Number_sound_1 = open("list_Number_sound.txt", "r")
data_1 = list_Number_sound_1.read()
list_Number_sound_1.close()
list_wavs_1 = data_1.split('\n')
del list_wavs_1[-1]

# opening the list with the results
list_answers_1 = open("answers.txt", "r")
data_1 = list_answers_1.read()
list_answers_1.close()
list_answers_1 = data_1.split('\n')
del list_answers_1[-1]

#%% Second 3min
# opening the list with the order in which the *.wav are played
list_Number_sound_2 = open("list_Number_sound_plus_one.txt", "r")
data_2 = list_Number_sound_2.read()
list_Number_sound_2.close()
list_wavs_2 = data_2.split('\n')
del list_wavs_2[-1]

# opening the list with the results
list_answers_2 = open("answers_plus_one.txt", "r")
data_2 = list_answers_2.read()
list_answers_2.close()
list_answers_2 = data_2.split('\n')
del list_answers_2[-1]


#%%

while True:
    # normal (first 3 min)
    if keyboard.is_pressed("1"):
        start = time.time()
        print(start)
        for number, answer in zip(list_wavs_1, list_answers_1):
            audio = os.path.join(path_wav + number)
            playsound(audio)
            print("Result:", answer, "; Number: ", number)
            time.sleep(2.5)  # Sleep for 2.5 seconds
            if keyboard.is_pressed("c"):
                break
        print("Result:", list_answers_1[-1], "; Number: ", number)  # last answer
        end = time.time()
        print(end - start)
        break
    # hard (second 3 min)
    if keyboard.is_pressed("2"):
        start = time.time()
        print(start)
        for number, answer in zip(list_wavs_2, list_answers_2):
            audio = os.path.join(path_wav + number)
            playsound(audio)
            print("Result:", answer, "; Number: ", number)
            time.sleep(2.5)  # Sleep for 2.5 seconds
            if keyboard.is_pressed("c"):
                break
        print("Result:", list_answers_2[-1], "; Number: ", number)  # last answer
        end = time.time()
        print(end - start)
        break

    if keyboard.is_pressed("c"):
        break
print("End")
