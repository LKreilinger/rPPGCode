import  threading
from functools import partial

from data_collection import PACED_test
from data_collection import RealsenseGetFrames
from multiprocessing import Process
if __name__ == '__main__':
    p1 = Process(target=PACED_test.paced_test())

    p2 = Process(target=RealsenseGetFrames.get_frames())
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    PACED_test.paced_test()
    RealsenseGetFrames.get_frames()
    #thread1 = threading.Thread(target=PACED_test.paced_test())
    #thread2 = threading.Thread(target=RealsenseGetFrames.get_frames())

# RealsenseGetFrames.get_frames()
#threading.Thread(target=partial (PACED_test.paced_test()).start()
#RealsenseGetFrames.get_frames()
