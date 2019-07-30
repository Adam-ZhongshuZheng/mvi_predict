from WrongListReader import *
import numpy as np
from matplotlib import pyplot as plt


def show_3D_npimg(img):
    data = np.load(img)
    print(data.shape)

    channels = data.shape[2]

    fig, ax = plt.subplots(1, channels)

    for x in range(channels):
        ax[x].imshow(data[:,:,x], cmap=plt.cm.gray)
        ax[x].set_title((x))
    fig.show()
    input()


def main():
    file = WrongListReader()
    for i in file.get_list():
        


if __name__ == '__main__':
	main()