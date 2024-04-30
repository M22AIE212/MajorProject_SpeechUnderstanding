import numpy as np
import matplotlib.pyplot as plt

def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()

def plotting(x_data, y_data, title, x_label, y_label):

    plt.figure(figsize=(20, 10))
    plt.plot(x_data, y_data)
    plt.title(title, fontsize=30)
    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label, fontsize=20)
    plt.show()
    

class Config():
    BASE_PATH = "/content/LibriSpeech/"
    training_dir = BASE_PATH + "/train-gram/"
    testing_dir = BASE_PATH + "/test-gram/"
    train_batch_size = 256
    train_number_epochs = 10
