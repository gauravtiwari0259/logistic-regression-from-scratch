prices=[4,9,1,10,8,3,12]

l = 0
r = 1
maxP = 0

while r < len(prices):
    if prices[l] < prices[r]:
        profit = prices[r] - prices[l]
        maxP = max(maxP, profit)
    else:
        l = r
    r += 1

print(maxP)





