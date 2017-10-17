#coding=utf-8
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
def hasCycle(head):
    if head == None:
        return False
    p1 = head
    p2 = head
    while p2.next and p2.next.next:
        p1 = p1.next
        p2 = p2.next.next
        if p1 == p2:
            return True
    return False
def detectCycle(self, head):
    if head==None or head.next==None:
        return None
    slow=head
    fast=head
    while fast.next and fast.next.next:
        slow=slow.next
        fast=fast.next.next
        if slow==fast:
            break
    if slow==fast:
        slow=head
        while slow!=fast:
            slow=slow.next
            fast=fast.next
        return slow
    return None

def reorderList(self, head):
    if head == None or head.next == None or head.next.next == None: return head
"""
    # break linked list into two equal length
    slow = fast = head  # 快慢指针技巧
    while fast and fast.next:  # 需要熟练掌握
        slow = slow.next  # 链表操作中常用
        fast = fast.next.next
    head1 = head
    head2 = slow.next
    slow.next = None

    # reverse linked list head2
    dummy = ListNode(0);
    dummy.next = head2  # 翻转前加一个头结点
    p = head2.next;
    head2.next = None  # 将p指向的节点一个一个插入到dummy后面
    while p:  # 就完成了链表的翻转
        tmp = p;
        p = p.next  # 运行时注意去掉中文注释
        tmp.next = dummy.next
        dummy.next = tmp
    head2 = dummy.next

    # merge two linked list head1 and head2
    p1 = head1;
    p2 = head2
    while p2:
        tmp1 = p1.next;
        tmp2 = p2.next
        p1.next = p2;
        p2.next = tmp1
        p1 = tmp1;
        p2 = tmp2
        """
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

def sortedArrayToBST(nums):
    """
    :type nums: List[int]
    :rtype: TreeNode
    """
    if len(nums)==0:
        return None
    res=TreeNode(0)
    mid_index=len(nums)/2
    mid=nums[mid_index]
    res.val=mid
    res.left=sortedArrayToBST(nums[:mid_index])
    res.right=sortedArrayToBST(nums[mid_index+1:])
    return res
def preorderTraversal(root):
    res=[]
    if root==None:
        return res
    tmp=[root]
    while len(tmp)!=0:
        p=tmp.pop()
        res.append(p.val)
        if p.right:
            tmp.append(p.right)
        if p.left:
            tmp.append(p.left)
    return res
# test=sortedArrayToBST([1,2,3,4,5,6,7])
# res=preorderTraversal(test)
# print res
def postorderTraversal(root):
    res=[]
    if root==None:
        return res
    tmp=[root]
    while len(tmp)!=0:
        p=tmp.pop()
        res.append(p.val)
        if p.left:
            tmp.append(p.left)
        if p.right:
            tmp.append(p.right)
    res.reverse()
    return res
# test=sortedArrayToBST([1,2,3,4,5,6,7])
# res=postorderTraversal(test)
# print res
import collections
class LRUCache:
    def __init__(self, capacity):
        """
        :type capacity: int
        """
        LRUCache.capacity=capacity
        LRUCache.length=0
        LRUCache.dict=collections.OrderedDict()

    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        try:
            value=LRUCache.dict[key]
            del LRUCache.dict[key]
            LRUCache.dict[key]=value
            return value
        except:
            return -1

    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: void
        """
        try:
            del LRUCache.dict[key]
            LRUCache.dict[key]=value
        except:
            if LRUCache.length==LRUCache.capacity:
                LRUCache.dict.popitem(last=False)
                LRUCache.length-=1
            LRUCache.dict[key]=value
            LRUCache.length+=1

    # # @param capacity, an integer
    # def __init__(self, capacity):
    #     LRUCache.capacity = capacity
    #     LRUCache.length = 0
    #     LRUCache.dict = collections.OrderedDict()
    #
    # # @return an integer
    # def get(self, key):
    #     try:
    #         value = LRUCache.dict[key]
    #         del LRUCache.dict[key]
    #         LRUCache.dict[key] = value
    #         return value
    #     except:
    #         return -1
    #
    # # @param key, an integer
    # # @param value, an integer
    # # @return nothing
    # def put(self, key, value):
    #     try:
    #         del LRUCache.dict[key]
    #         LRUCache.dict[key] = value
    #     except:
    #         if LRUCache.length == LRUCache.capacity:
    #             LRUCache.dict.popitem(last=False)
    #             LRUCache.length -= 1
    #         LRUCache.dict[key] = value
    #         LRUCache.length += 1
# cache=LRUCache(2)
#
# cache.put(1, 1)
# cache.put(2, 2)
# cache.get(1)
# cache.put(3, 3)
# cache.get(2)
# cache.put(4, 4)
# cache.get(1)
# cache.get(3)
# cache.get(4)
def insertionSortList(head):
    if head==None:
        return head
    tmp_head=ListNode(0)
    tmp_head.next=head
    curr=head
    while curr.next:
        if curr.next.val<curr.val:
            pre=tmp_head
            while pre.next.val<curr.next.val:
                pre=pre.next
            tmp=curr.next
            curr.next=curr.next.next
            tmp.next=pre.next
            pre.next=tmp
        else:
            curr=curr.next
    return tmp_head.next
# a,b,c,d,e=ListNode(1),ListNode(3),ListNode(4),ListNode(2),ListNode(5)
# a.next=b
# b.next=c
# c.next=d
# d.next=e
# res=insertionSortList(a)
# print res


def sortList(head):
    def merge(head1,head2):
        if head1==None:
            return head2
        if head2==None:
            return head1
        result=ListNode(0)
        p=result
        while head1 and head2:
            if head1.val<=head2.val:
                p.next=head1
                p=p.next
                head1=head1.next
            else:
                p.next=head2
                p=p.next
                head2=head2.next
        if head1==None:
            p.next=head2
        if head2==None:
            p.next=head1
        return result.next
    if head==None or head.next==None:
        return head
    slow=head;fast=head
    while fast.next and fast.next.next:
        slow=slow.next
        fast=fast.next.next
    head1=head
    head2=slow.next
    slow.next=None
    head1=sortList(head1)
    head2=sortList(head2)
    head=merge(head1,head2)
    return head
# a,b,c,d,e=ListNode(3),ListNode(1),ListNode(4),ListNode(5),ListNode(2)
# a.next=b
# b.next=c
# c.next=d
# d.next=e
# res=sortList(a)
# print res
import numpy as np
class Point(object):
    def __init__(self, a=0, b=0):
        self.x = a
        self.y = b
def maxPoints(points):
        """
        :type points: List[Point]
        :rtype: int
        """
        size_p=len(points)
        if size_p<3:
            return size_p
        result=-1

        for i in range(size_p):
            result_dict = {"inf": 0}
            samepoint=1
            for j in range(i+1,size_p):
                if points[i].x==points[j].x and points[i].y!=points[j].y:
                    result_dict['inf']+=1
                elif points[i].x!=points[j].x:
                    k = np.longdouble(1) * (points[i].y - points[j].y) / (points[i].x - points[j].x)
                    if k in result_dict:
                        result_dict[k]+=1
                    else:
                        result_dict[k]=1
                else:
                    samepoint+=1
            result=max(result,max(result_dict.values())+samepoint)
        return result
# test=[Point(0,0),Point(94911151,94911150),Point(94911152,94911151)]
# res=maxPoints(test)
# print res
def evalRPN(tokens):
    res=0
    tmp=[]
    for item in tokens:
        if item!='+' and item!='-' and item!='*' and item!='/':
            tmp.append(int(item))
        else:
            tmp_mid1=tmp.pop()
            tmp_mid2=tmp.pop()
            if item=='+':
                res=tmp_mid1+tmp_mid2
            elif item=='-':
                res=tmp_mid2-tmp_mid1
            elif item=='*':
                res=tmp_mid1*tmp_mid2
            else:
                if tmp_mid1*tmp_mid2>0:
                    res=tmp_mid2/tmp_mid1
                else:
                    res=-((-tmp_mid2)/tmp_mid1)
            tmp.append(res)
    res=tmp.pop()
    return res
# test= ["10","6","9","3","+","-11","*","/","*","17","+","5","+"]
# res=evalRPN(test)
# print res
def reverseWords(s):
    tmp=s.split()
    tmp.reverse()
    res=' '.join(tmp)
    return res
# test= "the sky is blue"
# res=reverseWords(test)
# print res
def maxProduct(nums):
    len_n=len(nums)
    res=[[0 for i in range(len_n)]for i in range(2)]
    res[0][0]=nums[0]
    res[1][0]=nums[0]
    for i in range(1,len_n):
        res[0][i]=max(res[0][i-1]*nums[i],res[1][i-1]*nums[i],nums[i])
        res[1][i] = min(res[0][i - 1] * nums[i], res[1][i - 1] * nums[i], nums[i])
    return max(res[0])
# test=[-2,3,-4]
# res=maxProduct(test)
# print res
def findMin(nums):
    len_n=len(nums)
    if len_n==1:
        return nums[0]
    left=0;right=len_n-1
    while left<right:
        mid=(right+left)/2
        if nums[mid]<nums[right]:
            right=mid
        else:
            left=mid+1
    return nums[left]
# test=[4, 5 ,6 ,7 ,0 ,1 ,2]
# res=findMin(test)
# print res


def findMin2(nums):
    len_n=len(nums)
    if len_n==1:
        return nums[0]
    left=0;right=len_n-1
    while left<right and nums[left]>=nums[right]:
        mid=(right+left)/2
        if nums[mid]<nums[right]:
            right=mid
        elif nums[mid]>nums[left]:
            left=mid+1
        else:
            left+=1
    return nums[left]
# test=[3,3,1,3]
# res=findMin2(test)
# print res
class MinStack(object):
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.list_nom=[]
        self.list_min=[]

    def push(self, x):
        """
        :type x: int
        :rtype: void
        """
        self.list_nom.append(x)
        if len(self.list_min)==0 or x<=self.list_min[-1]:
            self.list_min.append(x)
    def pop(self):
        """
        :rtype: void
        """
        tmp=self.list_nom.pop()
        if tmp==self.list_min[-1]:
            self.list_min.pop()
    def top(self):
        """
        :rtype: int
        """
        tmp=self.list_nom[-1]

        return tmp

    def getMin(self):
        """
        :rtype: int
        """
        return self.list_min[-1]
# minStack=MinStack()
# minStack.push(0);
# minStack.push(1);
# minStack.push(0);
# a=minStack.getMin();
# minStack.pop();
# d=minStack.getMin();
# print d
def getIntersectionNode(headA, headB):
    def getlen(head):
        length=0
        while head:
            head=head.next
            length+=1
        return length
    if headA==None or headB==None:
        return None
    lengthA=getlen(headA)
    lengthB=getlen(headB)
    if lengthA>lengthB:
        tmp=lengthA-lengthB
        while tmp>0:
            headA=headA.next
            tmp-=1
    if lengthA<lengthB:
        tmp=lengthB-lengthA
        while tmp>0:
            headB=headB.next
            tmp-=1
    while headA !=None and headB!=None:
        if headA==headB:
            return headA
        else:
            headB=headB.next
            headA=headA.next

    return None

# a1,b1,c1,d1,e1=ListNode(1),ListNode(3),ListNode(4),ListNode(2),ListNode(5)
# a1.next=b1
# b1.next=c1
# c1.next=d1
# d1.next=e1
# e1.next=None
# a2,b2,c2,d2,e2=ListNode(10),ListNode(11),ListNode(4),ListNode(2),ListNode(5)
# #a2.next=b2
# b2.next=c2
# c2.next=d2
# d2.next=e2
# e2.next=None
# res=getIntersectionNode(a1,b2)
# print res