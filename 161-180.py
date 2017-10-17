#coding=utf-8
def findPeakElement(nums):
    len_n=len(nums)
    if len_n<=1:
        return 0
    first=0;end=len_n-1
    while first<=end:
        if first==end:
            return first
        if first+1==end:
            return [first,end][nums[first]<nums[end]]
        mid=(first+end)/2
        if nums[mid]<nums[mid-1]:
            end=mid-1
        elif nums[mid]<nums[mid+1]:
            first=mid+1
        else:
            return mid
# test=[1,2,3,1]
# res=findPeakElement(test)
# print res
def maximumGap(nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    len_n=len(nums)
    if len_n<2:
        return 0
    min_value=min(nums)
    max_value=max(nums)
    bucket_range=max(1,(max_value-min_value-1)/(len_n-1)+1)
    bucket_len=(max_value-min_value)/bucket_range+1
    buckets=[None]*bucket_len
    for item in nums:
        bucket_idex=(item-min_value)/bucket_range
        bucket=buckets[bucket_idex]
        if bucket is None:
            bucket={'min':item,'max':item}
            buckets[bucket_idex]=bucket
        else:
            bucket['min']=min(bucket['min'],item)
            bucket['max']=max(bucket['max'],item)
    maxgap=buckets[0]['max']-buckets[0]['min']
    x=0
    while x <bucket_len:
        if buckets[x] is None :
            continue
        y=x+1
        while y<bucket_len and buckets[y] is None:
            y+=1
        if y<bucket_len:
            maxgap=max(maxgap,buckets[y]['min']-buckets[x]['max'])
        x=y
    return maxgap
import random
#test=[random.randint(0,100) for i in range(10)]
test1=[26, 69, 79, 74, 67, 5, 33, 41, 13, 46]
result=maximumGap(test1)
res=sorted(test1)
print result
