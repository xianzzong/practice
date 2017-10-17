def isSymmetric(self, root):
    def isSameTree(p, q):
        if p == None and q == None:
            return True
        if p and q:
            if p.val == q.val:
                return isSameTree(p.left, q.right) and isSameTree(p.right, q.left)
            else:
                return False
        else:
            return False

    if root:
        return isSameTree(root.left, root.right)
    else:
        return True
def levelOrder(self, root):
    """
    :type root: TreeNode
    :rtype: List[List[int]]
    """
    def solve(root,n):
        if root:
            if len(res)<n+1:
                res.append([])
            res[n].append(root.val)
            solve(root.left,n+1)
            solve(root.right,n+1)
    res=[]
    solve(root,0)
    return res
def zigzagLevelOrder(self, root):
    """
    :type root: TreeNode
    :rtype: List[List[int]]
    """
    def solve(root,n):
        if root:
            if len(res)<n+1:
                res.append([])
            if n%2==0:
                res[n].append(root.val)
            else:
                res[n].insert(0,root.val)
            solve(root.left,n+1)
            solve(root.right,n+1)
    res=[]
    solve(root,0)
    return res
def maxDepth(self, root):
    """
    :type root: TreeNode
    :rtype: int
    """
    maxdepth=0
    if root:
        maxdepth=max(self.maxDepth(root.left),self.maxDepth(root.right))+1
    return maxdepth
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
def buildTree(self, preorder, inorder):
    """
    :type preorder: List[int]
    :type inorder: List[int]
    :rtype: TreeNode
    """
    if len(preorder)==0:
        return None
    res=TreeNode(0)
    if len(preorder)==1:
        res.val=preorder[0]
        return res
    root_val=preorder[0]
    res.val=root_val
    root_index=inorder.index(root_val)
    res.left=self.buildTree(preorder[1:root_index+1],inorder[:root_index])
    res.right=self.buildTree(preorder[root_index+1:],inorder[root_index+1:])
    return res
def buildTree2(self, inorder, postorder):
    """
    :type inorder: List[int]
    :type postorder: List[int]
    :rtype: TreeNode
    """
    if len(inorder)==0:
        return None
    res=TreeNode(0)
    if len(inorder)==1:
        res.val=inorder[0]
        return res
    root_val=postorder[len(postorder)-1]
    res.val=root_val
    root_index=inorder.index(root_val)
    res.left=self.buildTree(inorder[:root_index],postorder[:root_index])
    res.right=self.buildTree(inorder[root_index+1:],postorder[root_index:len(postorder)-1])
    return res
def levelOrderBottom(self, root):
    """
    :type root: TreeNode
    :rtype: List[List[int]]
    """
    def solve(root,n):
        if root:
            if len(res)<n+1:
                res.append([])
            res[n].append(root.val)
            solve(root.left,n+1)
            solve(root.right,n+1)
    res=[]
    solve(root,0)
    res.reverse()

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
# test=[1,3]
# res=sortedArrayToBST(test)
# print (res)


def sortedListToBST(self, head):
    """
    :type head: ListNode
    :rtype: TreeNode
    """

    def solve(nums):
        if len(nums) == 0:
            return None
        res = TreeNode(0)
        mid_index = len(nums) / 2
        mid = nums[mid_index]
        res.val = mid
        res.left = solve(nums[:mid_index])
        res.right = solve(nums[mid_index + 1:])
        return res

    res = []
    while head:
        res.append(head.val)
        head = head.next
    ans = solve(res)
    return ans


def isBalanced(self, root):
    """
    :type root: TreeNode
    :rtype: bool
    """

    def getheight(root):
        if root:
            return max(getheight(root.left), getheight(root.right)) + 1
        else:
            return 0

    if root:
        if abs(getheight(root.left) - getheight(root.right)) > 1:
            return False
        else:
            return self.isBalanced(root.left) and self.isBalanced(root.right)
    else:
        return True
# test = [1, 2, 3, 4, 5, 6]
# tree = sortedArrayToBST(test)
# res=isBalanced(tree)

def minDepth(root):
    """
    :type root: TreeNode
    :rtype: int
    """
    if root:
        if root.left and root.right==None:
            return minDepth(root.left)+1
        elif root.right and root.left==None:
            return minDepth(root.right)+1
        else:
            return min(minDepth(root.left),minDepth(root.right))+1
    else:
        return 0
# test=sortedArrayToBST([2,1])
# res=minDepth(test)
# print res
def hasPathSum(root, sum):
    """
    :type root: TreeNode
    :type sum: int
    :rtype: bool
    """
    if root == None:
        return False
    tmp = sum - root.val
    if tmp == 0 and root.left == None and root.right == None:
        return True
    if hasPathSum(root.left, tmp) or hasPathSum(root.right, tmp):
        return True
    else:
        return False
# test=sortedArrayToBST([-2,-5])
# res=hasPathSum(test,-7)
# print res
def pathSum(root, sum):
    if root==None:
        return []
    tmp=sum-root.val
    res=[]
    if tmp==0:
        if root.left==None and root.right==None:
            return [[sum]]
        else:
            return []
    tmp_left=pathSum(root.left,tmp)
    tmp_right=pathSum(root.right,tmp)
    for item in tmp_left:
        res.append([root.val]+item)
    for item in tmp_right:
        res.append([root.val]+item)
    return res
# test=sortedArrayToBST([2,1])
# res=pathSum(test,1)
# print res
def pathSum(self, root, sum):
    """
    :type root: TreeNode
    :type sum: int
    :rtype: List[List[int]]
    """
    if root == None:
        return []
    tmp = sum - root.val
    res = []
    if root.left == None and root.right == None:
        if tmp == 0:
            return [[sum]]
        else:
            return []
    tmp_left = self.pathSum(root.left, tmp)
    tmp_right = self.pathSum(root.right, tmp)
    for item in tmp_left:
        res.append([root.val] + item)
    for item in tmp_right:
        res.append([root.val] + item)
    return res

def flatten(root):
    if root==None:
        return
    if root.left==None and root.right==None:
        return
    flatten(root.left);flatten(root.right)
    tmp=root.right
    root.right=root.left
    root.left=None
    while root.right:
        root=root.right
    root.right=tmp
# test=sortedArrayToBST([1,2,3,4,5,6])
# res=flatten(test)
# print res
def numDistinct(s, t):
        len_s=len(s);len_t=len(t)
        # if len_s==0 or len_t==0:
        #     return 0
        res=[[0 for i in range(len_t+1)] for j in range(len_s+1)]
        for row in range(len_s+1):
            res[row][0]=1
        for row in range(1,len_s+1):
            for col in range(1,len_t+1):
                if s[row-1]==t[col-1]:
                    res[row][col]=res[row-1][col-1]+res[row-1][col]
                else:
                    res[row][col]=res[row-1][col]
        return res[len_s][len_t]
# res=numDistinct('rabbbit','rabbit')
# print res
def connect(self, root):
    if root and root.left:
        root.left.next=root.right
        if root.next:
            root.right.next=root.next.left
        self.connect(root.left)
        self.connect(root.right)

def connect(root):
    if root:
        tmp_iteration=root
        tmp=None
        next_layer_begin=None
        while tmp_iteration:
            if tmp_iteration.left:
                if tmp:
                    tmp.next=tmp_iteration.left
                tmp=tmp_iteration.left
                if not next_layer_begin:
                    next_layer_begin=tmp_iteration.left
            if tmp_iteration.right:
                if tmp:
                    tmp.next=tmp_iteration.right
                tmp=tmp_iteration.right
                if not next_layer_begin:
                    next_layer_begin=tmp_iteration.right
            tmp_iteration=tmp_iteration.next
        connect(next_layer_begin)
def generate(numRows):
    """
    :type numRows: int
    :rtype: List[List[int]]
    """
    if numRows<=0:
        return []
    if numRows==1:
        return [[1]]
    if numRows==2:
        return [[1],[1,1]]
    res=[[1],[1,1]]
    for i in range(2,numRows):
        tmp=[];tmp.append(1)
        for j in range(1,i):
            tmp.append(res[i-1][j-1]+res[i-1][j])
        tmp.append(1)
        res.append(tmp)
    return res
# res=generate(5)
# print res
def getRow(rowIndex):
    if rowIndex<0:
        return []
    if rowIndex==0:
        return [1]
    if rowIndex==1:
        return [1,1]
    res=[1,1]
    for i in range(2,rowIndex+1):
        tmp=[];tmp.append(1)
        for j in range(1,i):
            tmp.append(res[j-1]+res[j])
        tmp.append(1)
        res=tmp[:]
    return res
# res=getRow(3)
# print res
def minimumTotal(triangle):
    len_t=len(triangle)
    if len_t==0:
        return 0
    res=[0 for i in range(len_t)]
    res[0]=triangle[0][0]
    for row in range(1,len_t):
        for col in range(len(triangle[row])-1,-1,-1):
            if col==len(triangle[row])-1:
                res[col]=res[col-1]+triangle[row][col]
            elif col==0:
                res[col]=res[col]+triangle[row][col]
            else:
                res[col]=min(res[col-1],res[col])+triangle[row][col]
    return min(res)
test=[
     [2],
    [3,4],
   [6,5,7],
  [4,1,8,3]
]
res=minimumTotal(test)
print res