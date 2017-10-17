def solve(test):
    max_val=max(test)
    min_val=min(test)
    judge=(max_val-min_val)/2
    tmp1=[];tmp2=[]
    min_index=test.index(min_val)
    tmp1.append(test[min_index])
    i=0
    while i<len(test):
        if i!=min_index:
            if abs(test[i]-test[min_index])<=judge:
                tmp1.append(test[i])
            else:
                tmp2.append(test[i])
        i+=1
    return tmp1,tmp2
test=[3,5,6,2,1]
res=solve(test)
print res