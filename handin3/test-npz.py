import numpy as np
a = np.arange(20).reshape((2, 10))
b = np.zeros((10, 1))
c = np.zeros((3, 4))

#np.savez('test-npz.npz', tzy=a, item2=b, test3=c)

#d = np.load('test-npz.npz')
#print(d.files)

e = np.zeros(5)
print(e)

f = np.zeros((5, 1))
print(f)


g = np.arange(100)
print(g[0:-1])
for i in range(0, len(g), 30):
    print(g[0+i:30+i])