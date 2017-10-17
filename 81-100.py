def search(nums, target):
    len_n=len(nums)
    if len_n==0:
        return False
    first=0;last=len_n-1
    while first<=last:
        mid=(first+last)/2
        print nums[mid]
        if nums[mid]==target:
            return True
        if nums[first]==nums[mid]==nums[last]:
            first+=1;last-=1
        elif nums[first]<=nums[mid]:
            if nums[mid]>target and nums[first]<=target:
                last=mid-1
            else:
                first=mid+1
        else:
            if nums[mid]<target and nums[last]>=target:
                first=mid+1
            else:
                last=mid-1
    return False
# test=[1]
# res=search(test,0)
# print res
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
def deleteDuplicates(head):
    if head==None:return head
    tmp=head
    while tmp.next !=None:
        if tmp.next.val==tmp.val:
            tmp.next=tmp.next.next
        else:
            tmp=tmp.next
    return head
def deleteDuplicates2(head):
    res=ListNode(0)
    res.next=head
    p=res;tmp=p.next
    while p.next:
        while tmp.next and tmp.next.val==p.next.val:
            tmp=tmp.next
        if tmp==p.next:
            p=p.next
            tmp=p.next
        else:
            p.next=tmp.next
    return res.next
# a,b,c,d,e=ListNode(1),ListNode(1),ListNode(1),ListNode(2),ListNode(3)
# a.next=b
# b.next=c
# c.next=d
# d.next=e
# res=deleteDuplicates2(a)
# print res
def largestRectangleArea(heights):
        maxarea=0
        length_list=[];idex_list=[]
        for i in range(len(heights)):
            if len(length_list)==0 or heights[i]>length_list[len(length_list)-1]:
                length_list.append(heights[i])
                idex_list.append(i)
            elif heights[i]<length_list[len(length_list)-1]:
                while len(length_list)>0 and heights[i]<length_list[len(length_list)-1]:
                    high=length_list.pop()
                    last_idex=idex_list.pop()
                    area=high*(i-last_idex)
                    if area>maxarea:
                        maxarea=area
                if len(length_list)==0 or heights[i]>length_list[len(length_list)-1]:
                    length_list.append(heights[i])
                    idex_list.append(last_idex)
        while len(length_list)>0:
            area=length_list.pop()*(len(heights)-idex_list.pop())
            if area>maxarea:
                maxarea=area
        return maxarea

# test=[2,1,2]
# res=largestRectangleArea(test)
# print res
def maximalRectangle(matrix):
    def largestarea(height):
        maxrec_row=0
        idex_list=[]
        i=0
        while i<len(height):
            if idex_list==[] or height[i]>height[idex_list[len(idex_list)-1]]:
                idex_list.append(i)
            else:
                curr=idex_list.pop()
                if idex_list==[]:
                    area=i*height[curr]
                else:
                    area=(i-idex_list[len(idex_list)-1]-1)*height[curr]
                maxrec_row=max(area,maxrec_row)
                i-=1
            i+=1
        while idex_list!=[]:
            curr=idex_list.pop()
            if idex_list == []:
                area = i * height[curr]
            else:
                area = (i - idex_list[len(idex_list) - 1] - 1) * height[curr]
            maxrec_row = max(area, maxrec_row)
        return maxrec_row
    row=len(matrix)
    if row==0:
        return 0
    col=len(matrix[0])
    if col==0:
        return 0
    max_area=0
    res=[0 for i in range(col)]
    for i in range(row):
        for j in range(col):
            if matrix[i][j]=='1':
                res[j]+=1
            else:
                res[j]=0
        area=largestarea(res)
        max_area=max(max_area,area)
    return max_area
# test=["10100",
#       "10111",
#       "11111",
#       "10010"]
# res=maximalRectangle(test)
# print (res)
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

def partition(head, x):
    first=ListNode(0)
    tmp_first=first
    last=ListNode(0)
    tmp_last=last
    while head:
        if head.val<x:
            first.next=head
            head=head.next
            first=first.next
            first.next=None
        else:
            last.next=head
            head=head.next
            last=last.next
            last.next=None
    first.next=tmp_last.next
    return tmp_first.next
# a,b,c,d,e=ListNode(2),ListNode(1),ListNode(3),ListNode(2),ListNode(5)
# a.next=b
# # b.next=c
# # c.next=d
# # d.next=e
# res=partition(a,2)
# print res
def isScramble(s1, s2):
    if len(s1)!=len(s2):
        return False
    l1=list(s1);l2=list(s2)
    l1.sort();l2.sort()
    if l1!=l2:return False
    if s1==s2:return True
    len_s=len(s1)
    for i in range(1,len_s):
        if isScramble(s1[:i],s2[:i]) and isScramble(s1[i:],s2[i:]):
            return True
        elif isScramble(s1[:i],s2[len_s-i:]) and isScramble(s1[i:],s2[:len_s-i]):
            return True
    return False
# s1='great'
# s2='rgeat'
# res=isScramble(s1,s2)
# print res
def merge(nums1, m, nums2, n):
    tmp=[];
    idex_1=0;idex_2=0
    while idex_1<m and idex_2<n:
        if nums1[idex_1]<=nums2[idex_2]:
            tmp.append(nums1[idex_1])
            idex_1+=1
        else:
            tmp.append(nums2[idex_2])
            idex_2+=1
    if idex_1!=m:
        while idex_1<m:
            tmp.append(nums1[idex_1])
            idex_1+=1
    if idex_2!=n:
        while idex_2<n:
            tmp.append(nums2[idex_2])
            idex_2+=1
    for i in range(m+n):
        if i<m:
            nums1[i]=tmp[i]
        else:
            nums1.append(tmp[i])
# test1=[1,3,5,7,10]
# test2=[2,3,4,6]
# merge(test1,5,test2,4)
# print test1
def grayCode(n):
    res=[]
    len_n=1<<n
    for i in range(len_n):
        tmp=(i>>1)^i
        res.append(tmp)
    return res
# res=grayCode(2)
# print (res)
def subsetsWithDup(nums):
    res=[]
    len_n=len(nums)
    if len_n==0:return [[]]
    nums.sort()
    tmp=subsetsWithDup(nums[:len_n-1])
    for item in tmp:
        if item not in res:
            res.append(item)
        tmp_mid=item+[nums[len_n-1]]
        if tmp_mid not in res:
            res.append(tmp_mid)
    return res
# test=[2,1,2]
# res=subsetsWithDup(test)
# print res
def numDecodings(s):
        len_s=len(s)
        if len_s==0 or s[0]=='0':
            return 0
        res=[1,1]
        for i in range(2,len_s+1):
            tmp_s=int(s[i-2:i])
            if 10<tmp_s<=26 and tmp_s!=20:
                res.append(res[i-2]+res[i-1])
            elif tmp_s==10 or tmp_s==20:
                res.append(res[i-2])
            elif s[i-1]!='0':
                res.append(res[i-1])
            else:
                return 0
        return res[len_s]
# test='1001'
# res=numDecodings(test)
# print res
def reverseBetween(head, m, n):
        if head==None or head.next==None:
            return head
        res=ListNode(0)
        t=res
        res.next=head
        for i in range(m-1):
            t=t.next
        tmp=t.next
        for i in range(n-m):
            tmp1=t.next
            t.next=tmp.next
            tmp.next=tmp.next.next
            t.next.next=tmp1
        return res.next
# a,b,c,d,e=ListNode(1),ListNode(2),ListNode(3),ListNode(4),ListNode(5)
# a.next=b
# b.next=c
# c.next=d
# d.next=e
# res=reverseBetween(a,2,4)
# print res
def restoreIpAddresses(s):
        def solve_ip(s,n):
            len_s=len(s)
            if len_s>3*n or (n==0 and len_s>0) or (n>0 and len_s<=0):
                return []
            if n==1:
                if len_s>1 and s[0]=='0':
                    return []
                elif int(s)<=255:
                    return [s]
                else:
                    return []
            i=1;res=[]
            if s[0]=='0':
                tmp=solve_ip(s[1:],n-1)
                for item in tmp:
                    res.append('0.'+item)
            else:
                while i<4:
                    tmp_str=s[:i]
                    if int(tmp_str)<=255:
                        tmp=solve_ip(s[i:],n-1)
                        if tmp != []:
                            for item in tmp:
                                res.append(tmp_str + '.' + item)
                    i+=1
            return res
        res=solve_ip(s,4)
        return res
# test="172162541"
# res=restoreIpAddresses(test)
# print res
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
def inorderTraversal(root):
    if root==None:
        return []
    res=inorderTraversal(root.left)+[root.val]+inorderTraversal(root.right)
    return res
# a,b,c=TreeNode(1),TreeNode(2),TreeNode(3)
# a.right=b
# b.left=c
# res=inorderTraversal(a)
# print (a)
def generateTrees(n):
    if n==0:
        return []
    def solve(begin,end):
        if begin>end:
            return [None]
        i=begin;res=[]
        while i<=end:
            tmp1=solve(begin,i-1)
            tmp2=solve(i+1,end)
            for item_1 in tmp1:
                for item_2 in tmp2:
                    tmp=TreeNode(i)
                    tmp.left=item_1
                    tmp.right=item_2
                    res.append(tmp)
            i+=1
        return res
    res=solve(1,n)
    return res
# res=generateTrees(3)
# print res
def numTrees(n):
    res=[1,1]
    for i in range(2,n+1):
        tmp=0
        if i%2==0:
            tmp_idex=i/2
            for j in range(tmp_idex):
                tmp=tmp+res[j]*res[i-1-j]
            tmp=tmp*2
            res.append(tmp)
        else:
            tmp_idex=i/2
            for j in range(tmp_idex):
                tmp=tmp+res[j]*res[i-1-j]
            tmp=tmp*2+res[tmp_idex]**2
            res.append(tmp)
    return res[n]
# res=numTrees(3)
# print (res)
def isInterleave(s1, s2, s3):
    len_s1=len(s1);len_s2=len(s2);len_s3=len(s3)
    if len_s1+len_s2!=len_s3:
        return False
    res=[[False for i in range(len_s2+1)] for i in range(len_s1+1)]
    res[0][0]=True
    for row in range(1,len_s1+1):
        if s1[row-1]==s3[row-1]:
            res[row][0]=True
        else:
            break
    for col in range(1,len_s2+1):
        if s2[col-1]==s3[col-1]:
            res[0][col]=True
        else:
            break
    for row in range(1,len_s1+1):
        for col in range(1,len_s2+1):
            if s1[row-1]==s3[row+col-1]:
                res[row][col]=res[row-1][col] or res[row][col]
            if s2[col-1]==s3[row+col-1]:
                res[row][col]=res[row][col-1] or res[row][col]
    return res[len_s1][len_s2]
# s1 = "aa"
# s2 = "ab"
# s3 = "aaba"
# res=isInterleave(s1,s2,s3)
# print (res)
class Solution(object):
    def getmin_max(self,root):
        if root==None:
            return [0,0]
        if root.left!=None:
            min_val=self.getmin_max(root.left)[0]
        else:
            min_val=root.val
        if root.right!=None:
            max_val=self.getmin_max(root.right)[1]
        else:
            max_val=root.val
        return [min_val,max_val]
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if root==None:
            return True
        if root.left!=None and self.getmin_max(root.left)[1]>=root.val:
            return False
        if root.right!=None and self.getmin_max(root.right)[0]<=root.val:
            return False
        if self.isValidBST(root.left) and self.isValidBST(root.right):
            return True
        return False
def recoverTree(self, root):
    """
    :type root: TreeNode
    :rtype: void Do not return anything, modify root in-place instead.
    """
    def getlist(root,list_r,list_p):
        if root:
            getlist(root.left,list_r,list_p)
            list_r.append(root.val)
            list_p.append(root)
            getlist(root.right,list_r,list_p)
    list_r,list_p=[],[]
    getlist(root,list_r,list_p)
    list_r.sort()
    for i in range(len(list_r)):
        list_p[i].val=list_r[i]
def isSameTree(self, p, q):
    """
    :type p: TreeNode
    :type q: TreeNode
    :rtype: bool
    """
    if p==None and q==None:
        return True
    if p and q:
        if p.val==q.val:
            return self.isSameTree(p.left,q.left) and self.isSameTree(p.right,q.right)
        else:
            return False
    else:
        return False