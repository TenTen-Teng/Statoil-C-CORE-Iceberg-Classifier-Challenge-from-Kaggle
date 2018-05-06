import caffe
import matplotlib.pyplot as plt
import numpy as np

caffe.set_mode_gpu()

solver = caffe. AdamSolver('solver.prototxt')

niter = 30000
test_interval = 100
train_loss = np.zeros(niter)
test_acc = np.zeros(int(np.ceil(niter / test_interval)))

# the main solver loop
for it in range(niter):
    solver.step(1) 
    # store the train loss
    train_loss[it] = solver.net.blobs['loss'].data
    solver.test_nets[0].forward(start='conv1')

    if it % test_interval == 0:
        acc=solver.test_nets[0].blobs['accuracy'].data
        print 'Iteration', it, 'testing...','accuracy:',acc
        test_acc[it // test_interval] = acc

    if it % test_interval == 0:
        test_error = solver.test_nets[0].blobs['loss'].data
        print 'Iteration', it, 'testing...','error:',test_error


print "average test error: ", np.mean(train_loss)

#----------------------------------------------------------------------------------------------
###########################Plotting Intermediate Layers, Weight################################
#---------------------------------------Define Functions---------------------------------------

def vis_square_f(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
                (0, 1), (0, 1))  # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.imshow(data,cmap='Greys',interpolation='nearest');
    plt.axis('off')
#----------------------------------------------------------------------------------------------
#------------------------------Plot All Feature maps Functions---------------------------------
plt.figure(1)
plt.semilogy(np.arange(niter), train_loss)
plt.xlabel('Number of Iteration')
plt.ylabel('Training Loss Values')
plt.title('Training Loss')

plt.figure(2)
plt.plot(test_interval * np.arange(len(test_acc)), test_acc, 'r')
plt.xlabel('Number of Iteration')
plt.ylabel('Test Accuracy Values')
plt.title('Test Accuracy')

plt.show()