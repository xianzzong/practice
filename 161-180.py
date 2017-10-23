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
# import random
# #test=[random.randint(0,100) for i in range(10)]
# test1=[26, 69, 79, 74, 67, 5, 33, 41, 13, 46]
# result=maximumGap(test1)
# res=sorted(test1)
# print result
def compareVersion(version1, version2):
    """
    :type version1: str
    :type version2: str
    :rtype: int
    """
    v1=version1.split('.')
    v2=version2.split('.')
    len_1=len(v1);len_2=len(v2)
    length=min(len_1,len_2)
    i=0
    while i<length:
        tmp1=int(v1[i])
        tmp2=int(v2[i])
        if tmp1==tmp2:
            i+=1
        elif tmp1<tmp2:
            return -1
        else:
            return 1
    if i==len_1 and i!=len_2:
        while i<len_2:
            if int(v2[i])>0:
                return -1
            i+=1
    elif i!=len_1 and i==len_2:
        while i<len_1:
            if int(v1[i])>0:
                return 1
            i+=1
    return 0
# test=['1.0','1']
# res=compareVersion(test[0],test[1])
# print res
def fractionToDecimal(numerator, denominator):
    if (numerator>=0 and denominator>0) or (numerator<=0 and denominator<0):
        result=''
    else:
        result='-'
    numerator=abs(numerator);denominator=abs(denominator)
    tmp_quotient=numerator/denominator
    tmp_remain=numerator%denominator
    result=result+str(tmp_quotient)
    quotient=[]
    remain=[]
    remain_index=-1
    while tmp_remain:
        remain.append(str(tmp_remain))
        tmp=tmp_remain*10
        tmp_quotient=tmp/denominator
        tmp_remain=tmp%denominator
        quotient.append(str(tmp_quotient))
        if (str(tmp_remain) in remain):
            remain_index=remain.index(str(tmp_remain))
            break
    if len(quotient) and remain_index!=-1:
        result=result+'.'+''.join(quotient[0:remain_index])+'('+''.join(quotient[remain_index:])+')'
    elif len(quotient) and remain_index==-1:
        result=result+'.'+''.join(quotient)
    return result
# res=fractionToDecimal(20,5)
# print res
def twoSum(numbers, target):
    len_n=len(numbers)
    left=0;right=len_n-1
    while left<right:
        if numbers[left]+numbers[right]>target:
            right-=1
        elif numbers[left]+numbers[right]<target:
            left+=1
        else:
            break
    if left==right:
        return 0
    return [left+1,right+1]
# test=[2,3,4]
# result=twoSum(test,6)
# print result
def convertToTitle(n):
    res=''
    while n:
        res=chr(ord('A')+(n-1)%26)+res
        n=(n-1)/26
    return res
# res=convertToTitle(298)
# print res
def majorityElement(nums):
    len_n=len(nums)
    result=nums[0]
    res_num=1
    for i in range(1,len_n):
        if res_num==0:
            result=nums[i]
            res_num=1
        else:
            if nums[i]==result:
                res_num+=1
            else:
                res_num-=1
    return result
# test=[1,2,1,2,3,1,1]
# res=majorityElement(test)
# print res
def titleToNumber(s):
    res=0
    for item in s:
        res=res*26+ord(item)-ord('A')+1
    return res
# res=titleToNumber(convertToTitle(256))
# print res
def trailingZeroes(n):
    res=0
    while n>=5:
        res+=n/5
        n=n/5
    return res
# res=trailingZeroes(30)
# print res
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class BSTIterator(object):
    def __init__(self, root):
        """
        :type root: TreeNode
        """
        self.result=[]
        self.putleft(root)
    def hasNext(self):
        """
        :rtype: bool
        """
        self.result
    def next(self):
        """
        :rtype: int
        """
        res=self.result.pop()
        self.putleft(res.right)
        return res.val
    def putleft(self,node):
        while node:
            self.result.append(node)
            node=node.left
def calculateMinimumHP(dungeon):
    row=len(dungeon)
    if row==0:
        return 0
    col=len(dungeon[0])
    if col==0:
        return 0
    result=[[0 for i in range(col)] for i in range(row)]
    result[row-1][col-1]=max(0,-dungeon[row-1][col-1])+1
    for row_index in range(row-1,-1,-1):
        for col_index in range(col-1,-1,-1):
            down=0
            if row_index+1<row:
                down=max(1,result[row_index+1][col_index]-dungeon[row_index][col_index])
            right=0
            if col_index+1<col:
                right=max(1,result[row_index][col_index+1]-dungeon[row_index][col_index])
            if down and right:
                result[row_index][col_index]=min(down,right)
            elif down:
                result[row_index][col_index]=down
            elif right:
                result[row_index][col_index] = right
    return result[0][0]
# test=[[-2,-3,3],[-5,-10,1],[10,30,-5]]
# #test=[[1,-3,3],[0,-2,0],[-3,-3,-3]]
# res=calculateMinimumHP(test)
# print res
def largestNumber(nums):
    def compare(a,b):
        if (a+b)>(b+a):
            return -1
        else:
            return 1
    res=sorted([str(item) for item in nums],cmp=compare)
    res=''.join(res).lstrip('0')
    if res=='':
        return '0'
    return res
test=[0,0]
res=largestNumber(test)
print res