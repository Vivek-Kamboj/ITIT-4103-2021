import numpy as np
import matplotlib.pyplot as plt
rand=np.random
fig,ax=plt.subplots(nrows=1,ncols=3,figsize=(12,7))
input1=rand.normal(8,1.5,20)
input2=rand.normal(8,1.5,20)
output= rand.binomial(100,0.6,20)
plt.subplot(1,3,1)
plt.title('input1')
plt.hist(input1,bins=50,color=’green’)
plt.subplot(1,3,2)
plt.title('input2')
plt.hist(input2,bins=50,color=’blue’)
plt.subplot(1,3,3)
plt.title('output')
plt.hist(output,bins=50,color=’gray’)
plt.show()
