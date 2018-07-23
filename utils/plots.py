from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

def plot_au(img, aus, title=None):
    '''
    Plot action units
    :param img: HxWx3
    :param aus: N
    :return:
    '''
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('off')
    fig.subplots_adjust(0, 0, 0.8, 1)  # get rid of margins

    # display img
    ax.imshow(img)

    if len(aus) == 11:
        au_ids = ['1','2','4','5','6','9','12','17','20','25','26']
        x = 0.1
        y = 0.39
        i = 0
        for au, id in zip(aus, au_ids):
            if id == '9':
                x = 0.5
                y -= .15
                i = 0
            elif id == '12':
                x = 0.1
                y -= .15
                i = 0

            ax.text(x + i * 0.2, y, id, horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, color='r', fontsize=20)
            ax.text((x-0.001)+i*0.2, y-0.07, au, horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, color='b', fontsize=20)
            i+=1

    else:
        au_ids = ['1', '2', '4', '5', '6', '7', '9', '10', '12', '14', '15', '17', '20', '23', '25', '26', '45']
        x = 0.1
        y = 0.39
        i = 0
        for au, id in zip(aus, au_ids):
            if id == '9' or id == '20':
                x = 0.1
                y -= .15
                i = 0

            ax.text(x + i * 0.2, y, id, horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, color='r', fontsize=20)
            ax.text((x-0.001)+i*0.2, y-0.07, au, horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, color='b', fontsize=20)
            i+=1

    if title is not None:
        ax.text(0.5, 0.95, title, horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, color='r', fontsize=20)

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return data