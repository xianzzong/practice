def firstMissingPositive(nums):
        len_n = len(nums)
        i = 0
        while i<len_n:
            if nums[i]>0 and nums[i]<len_n and nums[i]!=i+1 and nums[i]!=nums[nums[i]-1]:
                tmp=nums[i]-1
                nums[i],nums[tmp]=nums[tmp],nums[i]
            else:
                i+=1
        for i in range(len_n):
            if nums[i]!=i+1:
                return i+1
        return len_n+1
# test=[1,2,0]
# res=firstMissingPositive(test)
# print (res)
def trap(height):
        len_h=len(height)
        left_highest=[]
        leftmax=0
        for i in range(len_h):
            if height[i]>leftmax:
                leftmax=height[i]
                left_highest.append(leftmax)
            else:
                left_highest.append(leftmax)
        i=len_h-1
        right_max=0
        res=0
        while i>=0:
            if height[i]>right_max:
                right_max=height[i]
            res+=min(left_highest[i],right_max)-height[i]
            i-=1
        return res
# test=[0,1,0,2,1,0,1,3,2,1,2,1]
# res=trap(test)
# print res
def multiply(num1, num2):
        num1=num1[::-1]
        num2=num2[::-1]
        tmp_sum=[0 for i in range(len(num1+num2))]
        for i in range(len(num1)):
            for j in range(len(num2)):
                tmp_sum[i+j]+=int(num1[i])*int(num2[j])
        i=0
        res=[]
        tmp_add=0
        while i<len(num1)+len(num2):
            tmp_item=tmp_sum[i]+tmp_add
            mid=tmp_item%10
            res.append(str(mid))
            tmp_add=tmp_item/10
            i+=1
        i=len(num1)+len(num2)-1
        while i>0:
            if res[i]=='0':
                res.pop()
            else:
                break
            i-=1
        return ''.join(res)[::-1]
# a1='9133'
# a2='0'
# res=multiply(a1,a2)
# print res
def isMatch(s, p):
        len_s=len(s)
        len_p=len(p)
        flag=-1;s_point=0;p_point=0
        s_point_tmp=0
        while s_point<len_s:
            if p_point<len_p and (s[s_point]==p[p_point] or p[p_point]=='?'):
                s_point+=1;p_point+=1
                continue
            if p_point<len_p and p[p_point]=='*':
                flag=p_point;p_point+=1;s_point_tmp=s_point
                continue
            # if p_point==len_p:
            #     return True
            if flag!=-1:
                p_point=flag+1;s_point_tmp+=1;s_point=s_point_tmp
                continue
            return False
        while p_point<len_p:
            if p[p_point]!='*':
                return False
            p_point+=1
        if p_point==len_p:
            return True
        return False
# test1=''
# test2='*'
# res=isMatch(test1,test2)
# print res
def jump(nums):
        len_n=len(nums)
        current=0;last=0;res=0
        for i in range(len_n):
            if i>last:
                last=current
                res+=1
            current=max(current,i+nums[i])
        return res
# test=[2,3,1,1,4]
# res=jump(test)
# print res
def permute( nums):
    len_n=len(nums)
    if len_n==0:
        return []
    if len_n==1:
        return [nums]
    res=[]
    for i in range(len_n):
        tmp=permute(nums[:i]+nums[i+1:])
        for j in tmp:
            res.append([nums[i]]+j)
    return res
# test=[1,1,2]
# res=permute(test)
# print (res)
def permuteUnique(nums):
        res=[]
        len_n=len(nums)
        if len_n==0:
            return []
        if len_n==1:
            return [nums]
        i=0
        nums.sort()
        while i<len_n:
            if i>0:
                if nums[i]==nums[i-1]:
                    i+=1
                    continue
            tmp=permuteUnique(nums[:i]+nums[i+1:])
            for j in tmp:
                res.append([nums[i]]+j)
            i+=1
        return res
# test=[2,1,1]
# res=permuteUnique(test)
# print (res)
def rotate(matrix):
        N=len(matrix[0])
        for i in range(N):
            for j in range(i+1,N):
                matrix[i][j],matrix[j][i]=matrix[j][i],matrix[i][j]
        for i in range(N):
            matrix[i].reverse()
        return matrix
# test=[
#   [1,2,3],
#   [4,5,6],
#   [7,8,9]
# ]
# res=rotate(test)
# print res
def groupAnagrams(strs):
        strs_dict={}
        res=[]
        for item in strs:
            tmp=''.join(sorted(item))
            if tmp not in strs_dict:
                strs_dict[tmp]=[item]
            else:
                strs_dict[tmp]+=[item]
        for key in strs_dict.keys():
            res.append(strs_dict[key])
        return res
# test=["eat", "tea", "tan", "ate", "nat", "bat"]
# res=groupAnagrams(test)
# print (res)
def myPow(x, n):
        if n==0:
            return 1
        if n==1:
            return x
        if n<0:
            return 1/myPow(x,-n)
        if n%2==0:
            tmp=myPow(x,n/2)
            return tmp*tmp
        else:
            tmp=myPow(x,n/2)
            return tmp*tmp*x
# test=[8.88023,3]
# res=myPow(test[0],test[1])
# print res
def solveNQueens(n):
    def isValid(row,j):
        for i in range(row):
            if board[i]==j or abs(row-i)==abs(j-board[i]):
                return False
        return True
    def fillQ(row,row_val):
        if row==n:
            total_num.append(1)
            res.append(row_val)
            return
        for j in range(n):
            if isValid(row,j):
                board[row]=j
                fillQ(row+1,row_val+['.'*j+'Q'+'.'*(n-1-j)])
    board=[-1 for i in range(n)]
    res=[]
    total_num=[]
    fillQ(0,[])
    return total_num,res
# res=solveNQueens(4)
# print
def maxSubArray(nums):
        max_sum=nums[0]
        curr_sum=0
        for i in range(len(nums)):
            if curr_sum<0:
                curr_sum=0
            curr_sum=curr_sum+nums[i]
            max_sum=max(curr_sum,max_sum)
        return max_sum
# test=[-2,-1,-3,-4,-1,-2,-1,-5,-4]
# res=maxSubArray(test)
# print res
def spiralOrder(matrix):
        res=[]
        if len(matrix)==0:
            return res
        go_right=0;go_down=0
        go_left=len(matrix[0])-1;go_up=len(matrix)-1
        direct=0     #0 is goright,1 is go down,2 is go left,3 is go up
        while True:
            if direct==0:
                for i in range(go_right,go_left+1):
                    res.append(matrix[go_right][i])
                go_down+=1
            if direct==1:
                for i in range(go_down,go_up+1):
                    res.append(matrix[i][go_left])
                go_left-=1
            if direct==2:
                for i in range(go_left,go_right-1,-1):
                    res.append(matrix[go_up][i])
                go_up-=1
            if direct==3:
                for i in range(go_up,go_down-1,-1):
                    res.append(matrix[i][go_right])
                go_right+=1
            direct=(direct+1)%4
            if go_right>go_left or go_down>go_up:
                return res
        return res
# test=[
#  [ 4],
#  [ 2],
# ]
# res=spiralOrder(test)
# print res
def canJump(nums):
        current=0
        last=0;i=0
        if len(nums)<=1:
            return True
        for i in range(len(nums)):
            if nums[i]==0 and current<=i and i!=len(nums)-1:
                return False
            if i>last:
                last=current
            current=max(current,i+nums[i])
        if last>=len(nums)-1:
            return True
        else:
            return False
# test=[2,0,0]
# res=canJump(test)
# print (res)
class Interval(object):
    def __init__(self, s=0, e=0):
        self.start = s
        self.end = e
def merge(intervals):
    len_i=len(intervals)
    if len_i<=1:
        return intervals
    intervals.sort(key=lambda x:x.start)
    tmp=intervals[0]
    res=[]
    for item in intervals:
        if item.start<=tmp.end:
            tmp.end=max(item.end,tmp.end)
        else:
            res.append(tmp)
            tmp=item
    res.append(tmp)
    return res
# list_interval=[]
# a,b,c,d,e,f=Interval(),Interval(),Interval(),Interval(),Interval(),Interval()
# a.start=2;a.end=3
# b.start=5;b.end=5
# c.start=6;c.end=6
# d.start=7;d.end=7
# e.start=8;e.end=11
# f.start=6;f.end=13
# list_interval.append(a)
# list_interval.append(b)
# list_interval.append(c)
# list_interval.append(d)
# list_interval.append(e)
# res=merge(list_interval)
# print res
def insert(intervals, newInterval):
        len_i=len(intervals)
        res=[]
        if len_i==0:
            return [newInterval]
        left=0;right=len_i
        insert_position=-1
        while left<right:
            mid=(left+right)/2
            if intervals[mid].start==newInterval.start:
                insert_position=mid
                break
            if intervals[mid].start>newInterval.start:
                right=mid
            else:
                left=mid+1
        if insert_position==-1:
            insert_position=left
        intervals.insert(insert_position,newInterval)
        res=[];tmp=intervals[0]
        for item in intervals:
            if item.start<=tmp.end:
                tmp.end=max(item.end,tmp.end)
            else:
                res.append(tmp)
                tmp=item
        res.append(tmp)
        return res
# res=insert(list_interval,f)
# print (res)
def lengthOfLastWord(s):
        len_s=len(s)
        if len_s==0:
            return 0
        tmp_s=s.split(' ')
        i=len(tmp_s)-1
        while i>=0:
            if tmp_s[i]!='':
                return len(tmp_s[i])
            i-=1
        return 0
# test=' '
# res=lengthOfLastWord(test)
# print res
def generateMatrix(n):
    if n==0:
        return []
    matrix=[[0 for i in range(n)] for j in range(n)]
    go_left=0;go_down=0
    go_right=n-1;go_up=n-1
    direct=0   # 0 is go left ; 1 is go down; 2 is go right;3 is go up
    count=0
    while True:
        if direct==0:
            for i in range(go_left,go_right+1):
                count+=1
                matrix[go_down][i]=count
            go_down+=1
        if direct==1:
            for i in range(go_down,go_up+1):
                count+=1
                matrix[i][go_right]=count
            go_right-=1
        if direct==2:
            for i in range(go_right,go_left-1,-1):
                count+=1
                matrix[go_up][i]=count
            go_up-=1
        if direct==3:
            for i in range(go_up,go_down-1,-1):
                count+=1
                matrix[i][go_left]=count
            go_left+=1
        direct=(direct+1)%4
        if count==n*n:
            return matrix
# res=generateMatrix(7)
# print res
def getPermutation(n, k):
        nums=[1,2,3,4,5,6,7,8,9]
        nums=nums[:n]
        f_n_1=1;tmp_n=n
        res=''
        for i in range(1,n):
            f_n_1=f_n_1*i
        k=k-1
        for i in range(len(nums)-1,-1,-1):
            current=nums[k/f_n_1]
            res=res+str(current)
            nums.remove(current)
            if i!=0:
                k=k%f_n_1
                f_n_1=f_n_1/i
        return res
test=[6,400]
res=getPermutation(test[0],test[1])
print res