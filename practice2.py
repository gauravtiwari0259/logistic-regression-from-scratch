k=0
n=0
sum=0
nums = [2,7,11,15]
target=9
output=[]
for i in nums :
    k=k+1
    for j in nums:
            n=n+1
            if i==j:
                continue
            else:
                if sum == target:
                    print ("yes")
                    output.append(k-1)
                    output.append(n-1)
                    print(output)
                    break
                else:
                    continue
print(output)