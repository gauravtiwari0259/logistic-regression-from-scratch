nums = [2, 7, 2, 15]
target = 4
found = False
k = 0
l = 0
output = []

for i in nums:
    k = k + 1
    l = 0
    for j in nums:
        l = l + 1
        if i ==j and k == l:
            continue
        if i + j != target:
            continue
        else:
            sum = i + j
            if sum == target:
                found = True
                output.append(k - 1)
                output.append(l - 1)
                print(output)
                break
    if found:
        break

