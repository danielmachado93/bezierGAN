import numpy as np
from utils import*
from glob import glob
import os
i = 50
batch_size = 10
N = 256

out_width = 128
out_height = 128


t = np.linspace(0.0,1.0)
t = np.expand_dims(t,axis=1)

I1 = np.ones_like(t)
I2 = t
I3 = t*t
I4 = t*t*t

I = np.concatenate((I1,I2,I3,I4), axis=1) # [i,4] parametric space
It = np.array([[1,0,0,0],[-3,3,0,0],[3,-6,3,0],[-1,3,-3,1]],dtype=np.float32) # [4,4] cubic Berbstein polynomials coeffients
I = np.matmul(I,It)

# Repeat for all batch size
I = np.tile(I, (10,1,1))
BP = np.random.normal(loc=0.5, scale=0.02,size=(batch_size,4,2*N))

C = np.matmul(I,BP)
C = np.reshape(C, newshape=[batch_size,N*i,2])
C = np.multiply(C, np.array([out_width,out_height]))
C = np.floor(C).astype('int32')

# Define Data for training
data = glob(os.path.join("./data", 'celebA-8500', '*.jpg') )
image = get_image(data[0],128,128,128,128,True,True)
out = np.where(image < 0.0, 0.0, 1.0)


def func(input):
    m = np.array([[1.0,2.0,0.0],[3.0,4.0,0.0]])
    out = np.matmul(input,m)
    return out

input = np.random.normal(0.0,0.2,size=[10,50,2])
b = func(input)

h =np.ones(shape=[])

print(b.shape)
