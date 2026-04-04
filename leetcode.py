nums = [1,2,3,3,4]
x = nums
y=[]
left=0
right=len(x)-1
k=0
while k<len(x):
    if left<right:
        if x[left] != x[right]:
            right = right - 1
        elif x[left] == x[right]:
            y.append(1)
            break
    elif left == right:
        break
    k=k+1
print(len(y))
if len(y)==0:
    print("false")
else:
    print("true")



