# -*- coding: utf-8 -*-
"""
Created on Mon May  9 15:31:57 2022

@author: Chaputa
"""
# copy Folder Structure from data to output
 
import os
def copyFolderStructure(inputpath,outputpath):
    for dirnames in os.listdir(inputpath):
        structure = os.path.join(outputpath+'/'+dirnames)
        if not os.path.isdir(structure):
            os.mkdir(structure)
        else:
            print("Folder does already exits!")