import numpy as np
a = np.array([1,2,3,4,5])
print(a)

size=a.size
print(size)

dimension=a.ndim
print(dimension)
print(" ")
print(" ")
print(" ")

u=[1,0,2]
v=[2,3,4]
z=[]
for u,v in zip(u,v):
    z.append(u+v)
print(z)


x=np.array([1,2,3])
y=np.array([2,3,4])
k=x+y
print(k)

l=np.dot(x,y)
print(l)
print(" ")

print(np.linspace(-2,2,5))
integer_array = np.arange(-9, 9)
print(integer_array)
print(np.linspace(0,2*np.pi,100))