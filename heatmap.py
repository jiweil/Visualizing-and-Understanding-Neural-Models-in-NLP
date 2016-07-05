import numpy
import matplotlib.pylab as plt

matrix = numpy.loadtxt("matrix")
matrix_magnify=numpy.zeros((matrix.shape[0]*10,matrix.shape[1]))
for i in range(matrix.shape[0]):
    for j in range(10):
        matrix_magnify[i*10+j,:]=matrix[i,:]
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.imshow(matrix_magnify, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()
plt.show()
