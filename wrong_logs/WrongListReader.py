#######################################################################
#
#
#   The file for reading the wrong list and showing the wrong images.
#
#   By AdamMo,
#  30 July, 2019
#######################################################################

import os 
import pdb

class WrongListReader():

    def __init__(self, listpath='wrong_list7.adm', allfilepath='../data_flods/flod7/valid_set.csv'):
        self.listfile = listpath
        self.allfile = allfilepath
        self.imglist = []

    def get_list(self):
        """Get all the image message from wronglist

        Return:
            imglist: All the message about the wrong list; a unit in the list stands for an epoch's all wrong id imgs. 
                Every one in the unit is a dict for i img.
        """
        imglist = []    # for return
        alllist = {}    # all the data in the set
        
        with open(self.allfile, 'r') as allfile:
            for data_line in allfile.readlines():
                data_line = data_line.replace('\n', '').replace('\r', '').split(',')
                ids = int(data_line[0])
                label = int(data_line[1])
                filepath = data_line[2]
                if ids not in alllist:
                    alllist[ids] = {'id':ids, 'label': label, 'path': filepath}
        with open(self.listfile, 'r') as f:
            for data_line in f.readlines():
                data_line = data_line.replace('\n', '').replace('\r', '').split(',')
                epoch = data_line[0]
                acc = data_line[1]
                ids = data_line[2:-1]
                tmpdict = {}
                for i in ids:
                    tmpdict[i] = alllist[int(i)]

                imglist.append(tmpdict)
        self.imglist = imglist
        return imglist

    def 


if __name__ == '__main__':
    file = WrongListReader()
    for i in file.get_list():
        print(i)
