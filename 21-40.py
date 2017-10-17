class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
def mergeTwoLists(l1, l2):
        head=ListNode(0)
        tmp=head
        if l1==None and l2==None:
            return None
        while l1!=None or l2!=None:
            if l1==None:
                while l2!=None:
                    tmp.val=l2.val
                    l2=l2.next
                    if l2==None:
                        break
                    tmp.next=ListNode(0)
                    tmp=tmp.next
                break
            if l2==None:
                while l1!=None:
                    tmp.val=l1.val
                    l1=l1.next
                    if l1==None:
                        break
                    tmp.next=ListNode(0)
                    tmp=tmp.next
                break
            if l1.val<=l2.val:
                tmp.val=l1.val
                l1=l1.next
            else:
                tmp.val=l2.val
                l2=l2.next
            tmp.next=ListNode(0)
            tmp=tmp.next
        return head
def generateParenthesis(n):
        res=[]
        if n==0:
            return res
        if n==1:
            return ['()']
        for i in xrange(0,n):
            tmp1=generateParenthesis(i)
            tmp2=generateParenthesis(n-i-1)
            for item1 in tmp1:
                for item2 in tmp2:
                    res.append('('+item1+')'+item2)
                if len(tmp2)==0:
                    res.append('('+item1+')')
            if len(tmp1)==0:
                for item2 in tmp2:
                    res.append('()'+item2)
        return res
# test=3
# res=generateParenthesis(test)
# print res
def mergeTwoLists(self, l1, l2):
    head = ListNode(0)
    tmp = head
    if l1 == None and l2 == None:
        return None
    while l1 != None or l2 != None:
        if l1 == None:
            while l2 != None:
                tmp.val = l2.val
                l2 = l2.next
                if l2 == None:
                    break
                tmp.next = ListNode(0)
                tmp = tmp.next
            break
        if l2 == None:
            while l1 != None:
                tmp.val = l1.val
                l1 = l1.next
                if l1 == None:
                    break
                tmp.next = ListNode(0)
                tmp = tmp.next
            break
        if l1.val <= l2.val:
            tmp.val = l1.val
            l1 = l1.next
        else:
            tmp.val = l2.val
            l2 = l2.next
        tmp.next = ListNode(0)
        tmp = tmp.next
    return head


def mergeKLists(self, lists):
    """
    :type lists: List[ListNode]
    :rtype: ListNode
    """
    len_lists = len(lists)
    if len_lists == 0:
        return None
    if len_lists == 1:
        return lists[0]
    n = len_lists / 2
    tmp1 = self.mergeKLists(lists[:n])
    tmp2 = self.mergeKLists(lists[n:])
    return self.mergeTwoLists(tmp1, tmp2)
def swapPairs(head):
        if head==None:
            return None
        res=ListNode(0)
        res.next=head
        tmp=res
        while tmp.next!=None and tmp.next.next!=None:
            t=tmp.next.next
            tmp.next.next=t.next
            t.next=tmp.next
            tmp.next=t
            tmp=tmp.next.next
        return res.next
# a=ListNode(1);
# b=ListNode(2);
# c=ListNode(3);
# d=ListNode(4);
# a.next = b
# b.next=c
# c.next=d
# d.next=None
# res=swapPairs(a)
# print res
def reverse(start,end):
    nhead=ListNode(0)
    nhead.next=start
    while nhead.next!=end:
        tmp=start.next
        start.next=tmp.next
        tmp.next=nhead.next
        nhead.next=tmp
    return [end,start]
def reverseLGroup(head,k):
        res=ListNode(0)
        if head==None:
            return None
        res.next=head
        start=res
        while start.next!=None:
            end=start
            i=0
            while i<k-1:
                end=end.next
                if end.next==None:
                    return res.next
                i+=1
            tmp=reverse(start.next,end.next)
            start.next=tmp[0]
            start=tmp[1]
        return res.next
# a=ListNode(1);
# b=ListNode(2);
# c=ListNode(3);
# d=ListNode(4);
# a.next = b
# b.next=c
# c.next=d
# d.next=None
# res=reverseLGroup(a,3)
# print res
def removeDuplicates(nums):
        len_nums=len(nums)
        if len_nums==0:
            return None
        org=1
        res=1
        while org<len_nums:
            if nums[org]!=nums[org-1]:
                nums[res]=nums[org]
                org+=1
                res+=1
            else:
                org+=1
        return res
# a=[1,2]
# res=removeDuplicates(a)
# print res
def removeElement(nums,val):
        len_nums=len(nums)
        # if len_nums==0:
        #     return 0
        org=0;res=0
        while org<len_nums:
            if nums[org] != val:
                nums[res] = nums[org]
                org += 1
                res += 1
            else:
                org += 1
        return res
# test=[]
# val=3
# res=removeElement(test,val)
# print res
def strStr(haystack, needle):
        lenh=len(haystack)
        lenn=len(needle)
        if lenh<lenn:
            return -1
        if lenn==0:
            return 0
        i=0
        while i<=lenh-lenn:
            if haystack[i]!=needle[0]:
                i+=1
            else:
                j=0
                while j<lenn:
                    if haystack[i+j]==needle[j]:
                        j+=1
                    else:
                        i+=1
                        break
                if j==lenn:
                    return i
        return -1
# test=''
# need=''
# res=strStr(test,need)
# print res
def divide(dividend, divisor):
        flag=True
        if dividend>0 and divisor<0:
            flag=False
        if dividend<0 and divisor>0:
            flag=False
        dividend=abs(dividend)
        divisor=abs(divisor)
        if dividend<divisor:
            return 0
        tmp=divisor
        res=1
        while dividend>=tmp:
            tmp=tmp<<1
            if tmp>dividend:
                break
            res=res<<1
        tmp=tmp>>1
        rea=res+divide(dividend-tmp,divisor)
        if flag:
            if rea>2147483647:
                return 2147483647
            return rea
        if rea>2147483648:
            return -2147483648
        return -rea
# test=[7,2]
# res=divide(test[0],test[1])
# print res
def findSubstring(s, words):
        lens=len(s)
        num_words=len(words)
        res=[]
        if lens==0:
            return res
        dic_words={}
        for item in words:
            if item in dic_words:
                dic_words[item]+=1
            else:
                dic_words[item]=1
        j=0
        len_word=len(words[0])
        while j<=lens-num_words*len_word:
            tmp_d=dic_words.copy()
            tmp_s=s[j:j+num_words*len_word]
            i=0
            while i<=len(tmp_s)-len_word:
                tmp_s_word=tmp_s[i:i+len_word]
                if tmp_s_word in tmp_d and tmp_d[tmp_s_word]!=0:
                    tmp_d[tmp_s_word]-=1
                else:
                    break
                i+=len_word
            if i==len(tmp_s):
                res.append(j)
            j+=1
        return res
# test1='barfoothefoobarman'
# words=['foo','bar']
# res=findSubstring(test1,words)
# print res
def nextPermutation(nums):
        len_nums=len(nums)
        if len_nums<=1:
            return
        i=len_nums-2
        while i>=0:
            if nums[i]<nums[i+1]:
                j=i+1
                while j<len_nums:
                    if nums[i]>=nums[j]:
                        break
                    j+=1
                j-=1
                nums[i],nums[j]=nums[j],nums[i]
                nums[i+1:]=sorted(nums[i+1:])
                return
            i-=1
        mid=len_nums/2
        k=0
        while k<mid:
            nums[k],nums[len_nums-1-k]=nums[len_nums-1-k],nums[k]
            k+=1
        return
# test=range(100,-1,-1)
# nextPermutation(test)
# print test
def longestValidParentheses(s):
        len_s=len(s)
        res=0
        if len_s<=1:
            return res
        i=0;last=-1 #last is last  bu pi pei de ')'
        idex_p=[]
        for i in xrange(len_s):
            if s[i]=='(':
                idex_p.append(i)
            if s[i]==')':
                if len(idex_p)==0:
                    last=i
                else:
                    idex_p.pop()
                    if len(idex_p)==0:
                        res=max(res,i-last)
                    else:
                        res=max(res,i-idex_p[len(idex_p)-1])
        return res
# test='))))))()((())))'
# print len(test)
# res=longestValidParentheses(test)
# print res
def search(nums, target):
        len_nums=len(nums)
        if len_nums==0:
            return -1
        first=0;last=len_nums
        while first<last:
            midd=(first+last)/2
            if nums[midd]==target:
                return midd
            if nums[first]<nums[midd]:
                if nums[first]<=target and nums[midd]>target:
                    last=midd
                else:
                    first=midd+1
            else:
                if nums[midd]<target and nums[last-1]>=target:
                    first=midd+1
                else:
                    last=midd
        return -1
# test=[5,1,3]
# target=3
# res=search(test,target)
# print res
def findleft(nums,first,mid):
    if nums[first]==nums[mid]:
        return first
    middle=(first+mid)/2
    if nums[middle]==nums[mid]:
        return findleft(nums,first+1,middle)
    return findleft(nums,middle+1,mid)
def findright(nums,last,mid):
    if nums[last]==nums[mid]:
        return last
    middle=(mid+last)/2
    if nums[middle]==nums[mid]:
        return findright(nums,last-1,middle)
    return findright(nums,middle-1,mid)
def searchRange(nums, target):
        len_nums=len(nums)
        res=[-1,-1]
        if len_nums==0:
            return res
        left=0;right=len_nums-1
        while left<=right:
            mid=(left+right)/2
            if nums[mid]<target:
                left=mid+1
            if nums[mid]>target:
                right=mid-1
            if nums[mid]==target:
                res[0]=findleft(nums,left,mid)
                res[1]=findright(nums,right,mid)
                return res
        return res
# test=[1,2,3,3,3,3,4,5,9]
# target=3
# res=searchRange(test,target)
# print res
def searchInsert(nums, target):
        len_nums=len(nums)
        if len_nums==0:
            return 0
        left=0;right=len_nums-1
        while left<right:
            mid=(left+right)/2
            if nums[mid]==target:
                return mid
            if nums[mid]<target:
                left=mid+1
            if nums[mid]>target:
                right=mid-1
        if nums[left]<target:
            return left+1
        else:
            return left
# test=[1,3,5,6]
# res=searchInsert(test,7)
# print res
def isValidSudoku(board):
        # judge row
        for row in xrange(9):
            judge=[]
            for col in xrange(9):
                if board[row][col]!='.' and board[row][col] in judge:
                    return False
                else:
                    judge.append(board[row][col])
        #judge col
        for col in xrange(9):
            judge=[]
            for row in xrange(9):
                if board[row][col]!='.' and board[row][col] in judge:
                    return False
                else:
                    judge.append(board[row][col])
        row=0
        while row<9:
            col = 0
            while col<9:
                i=0;j=0
                judge=[]
                for i in xrange(3):
                    for j in xrange(3):
                        if board[row+i][col+j]!='.' and board[row+i][col+j] in judge:
                            return False
                        else:
                            judge.append(board[row+i][col+j])
                col+=3
            row+=3
        return True
test=[".87654321",
      "2........",
      "3........",
      "4........",
      "5........",
      "6........",
      "7........",
      "8........",
      "9........"]
# res=isValidSudoku(test)
# print res
def isVaild_xy(board,x,y):
    for i in xrange(9):
        if i!=x and board[i][y]==board[x][y]:
            return False
    for j in xrange(9):
        if j!=y and board[x][j]==board[x][y]:
            return False
    row_x=(x/3)*3;row_y=(y/3)*3
    for i in xrange(3):
        for j in xrange(3):
            if ((row_x+i)!=x or (row_y+j)!=y) and board[row_x+i][row_y+j]==board[x][y]:
                return False
    return True
def fillsudoku(board):
    for row in range(9):
        for col in range(9):
            if board[row][col]=='.':
                for k in '123456789':
                    #board[row][col]=k
                    board[row] = board[row][:col] + k + board[row][col + 1:]
                    if isVaild_xy(board,row,col) and fillsudoku(board):
                        return True
                    #board[row][col]='.'
                    board[row] = board[row][:col] + '.' + board[row][col + 1:]
                return False
    return True
# test=["..9748...",
#       "7........",
#       ".2.1.9...",
#       "..7...24.",
#       ".64.1.59.",
#       ".98...3..",
#       "...8.3.2.",
#       "........6",
#       "...2759.."]
# res=fillsudoku(test)
# print (test)
def countStr(stri):
    len_str=len(stri)
    res='';count=0;tmp=stri[0]
    for i in range(len_str):
        if stri[i]==tmp:
            count+=1
        else:
            res+=str(count)+tmp
            count=1
            tmp=stri[i]
    res+=str(count)+tmp
    return res
def countAndSay(n):
        res='1'
        for i in range(n-1):
            res=countStr(res)
        return res
# res=countAndSay(6)
# print res

def com_sum(candidates,target,j):
    result=[]
    len_c=len(candidates)
    if target==0:
        return []
    if len_c<j+1 or target<0:
        return [[-1]]
    tmp1=com_sum(candidates,target,j+1)
    tmp2=com_sum(candidates,target-candidates[j],j)
    if len(tmp2)==0:
        result.append([candidates[j]])
    elif tmp2!=[[-1]]:
        for item in tmp2:
            result.append([candidates[j]]+item)
    if len(tmp1)!=0 and tmp1!=[[-1]]:
        for item in tmp1:
            result.append(item)
    if tmp1==[[-1]] and tmp2==[[-1]]:
        return [[-1]]
    return result
def combinationSum(candidates, target):
        candidates.sort()
        res=com_sum(candidates,target,0)
        if res==[[-1]]:
            return []
        return res
# test=[2,3,6,7]
# res=combinationSum(test,7)
# print
def com_sum2(candidates,target,j):
    result=[]
    if target==0:
        return []
    len_c=len(candidates)
    if len_c<j+1 or target<0:
        return [[-1]]
    tmp2=com_sum2(candidates,target-candidates[j],j+1)
    n=1
    while j+n<len_c:
        if candidates[j+n]!=candidates[j]:
            break
        n+=1
    tmp1=com_sum2(candidates,target,j+n)
    if len(tmp2)==0:
        result.append([candidates[j]])
    elif tmp2!=[[-1]]:
        for item in tmp2:
            result.append([candidates[j]]+item)
    if tmp1!=[[-1]]:
        for item in tmp1:
            result.append(item)
    if tmp1==[[-1]] and tmp2==[[-1]]:
        return [[-1]]
    return result
def combinationSum2(candidates, target):
        candidates.sort()
        res=com_sum2(candidates,target,0)
        if res==[[-1]]:
            return []
        return res
test=[10, 1, 2, 7, 6, 1, 5]
res=combinationSum2(test,8)
print res