class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

# 101. Symmetric Tree


class Solution(object):
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        def judge(left, right):
            if left is None and right is None:
                return True
            if left and right:
                if left.val == right.val:
                    return judge(left.left, right.right) and judge(left.right, right.left)
                else:
                    return False
            else:
                return False
        if root is None:
            return True
        else:
            return judge(root.left, root.right)

# 102. Binary Tree Level Order Traversal


class Solution(object):
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        def solve(node, n):
            if node:
                if len(res) < n + 1:
                    res.append([])
                res[n].append(node.val)
                solve(node.left, n + 1)
                solve(node.right, n + 1)
            return
        res = []
        solve(root, 0)
        return res

# 103. Binary Tree Zigzag Level Order Traversal


class Solution(object):
    def zigzagLevelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        def solve(node, n):
            if node:
                if len(res) < n + 1:
                    res.append([])
                if n % 2 == 0:
                    res[n].append(node.val)
                else:
                    res[n].insert(0, node.val)
                solve(node.left, n + 1)
                solve(node.right, n + 1)
            return
        res = []
        solve(root, 0)
        return res

# 104. Maximum Depth of Binary Tree


class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        max_depth = 0
        if root:
            max_depth = max(self.maxDepth(root.left),
                            self.maxDepth(root.right)) + 1
        return max_depth

# 105. Construct Binary Tree from Preorder and Inorder Traversal


class Solution(object):
    def buildTree(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        res = TreeNode(0)
        if len(preorder) == 0:
            return None
        if len(preorder) == 1:
            res.val = preorder[0]
            return res
        root_value = preorder[0]
        res.val = root_value
        root_index = inorder.index(root_value)
        res.left = self.buildTree(
            preorder[1:root_index + 1], inorder[:root_index])
        res.right = self.buildTree(
            preorder[root_index + 1:], inorder[root_index + 1:])
        return res

# 106. Construct Binary Tree from Inorder and Postorder Traversal


class Solution(object):
    def buildTree(self, inorder, postorder):
        """
        :type inorder: List[int]
        :type postorder: List[int]
        :rtype: TreeNode
        """
        res = TreeNode(0)
        if len(inorder) == 0:
            return None
        if len(inorder) == 1:
            res.val = inorder[0]
            return res
        root_value = postorder[-1]
        res.val = root_value
        root_index = inorder.index(root_value)
        res.left = self.buildTree(inorder[:root_index], postorder[:root_index])
        res.right = self.buildTree(
            inorder[root_index + 1:], postorder[root_index:-1])
        return res

# 107. Binary Tree Level Order Traversal II


class Solution(object):
    def levelOrderBottom(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        def solve(node, n):
            if node:
                if len(res) < n + 1:
                    res.append([])
                res[n].append(node.val)
                solve(node.left, n + 1)
                solve(node.right, n + 1)
            return
        res = []
        solve(root, 0)
        res.reverse()
        return res


# 108. Convert Sorted Array to Binary Search Tree
class Solution(object):
    def sortedArrayToBST(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        if len(nums) == 0:
            return None
        res = TreeNode(0)
        mid_index = len(nums) / 2
        res.val = nums[mid_index]
        res.left = self.sortedArrayToBST(nums[:mid_index])
        res.right = self.sortedArrayToBST(nums[mid_index + 1:])
        return res

# 109. Convert Sorted List to Binary Search Tree


class Solution(object):
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
            res.val = nums[mid_index]
            res.left = solve(nums[:mid_index])
            res.right = solve(nums[mid_index + 1:])
            return res
        nums = []
        while head:
            nums.append(head.val)
            head = head.next
        res = solve(nums)
        return res

# 110. Balanced Binary Tree


class Solution(object):
    def isBalanced(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        def getdepth(node):
            if node is None:
                return 0
            left = getdepth(node.left)
            if left == -1:
                return -1
            right = getdepth(node.right)
            if right == -1:
                return -1
            diff = abs(left - right)
            if diff > 1:
                return -1
            return 1 + max(left, right)
        judge = getdepth(root)
        if judge == -1:
            return False
        else:
            return True

# 111. Minimum Depth of Binary Tree


class Solution(object):
    def minDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """

        if root is None:
            return 0
        else:
            if root.left and root.right:
                return 1 + min(self.minDepth(root.left), self.minDepth(root.right))
            elif root.left and root.right is None:
                return 1 + self.minDepth(root.left)
            elif root.right and root.left is None:
                return 1 + self.minDepth(root.right)
            else:
                return 1

# 112. Path Sum


class Solution(object):
    def hasPathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: bool
        """
        if root is None:
            return False
        tmp = sum - root.val
        if tmp == 0 and root.left is None and root.right is None:
            return True
        elif self.hasPathSum(root.left, tmp) or self.hasPathSum(root.right, tmp):
            return True
        else:
            return False


# 113. Path Sum II
class Solution(object):
    def pathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: List[List[int]]
        """
        res = []
        if root is None:
            return []
        tmp = sum - root.val
        if root.left is None and root.right is None:
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

# 114. Flatten Binary Tree to Linked List


class Solution(object):
    def flatten(self, root):
        """
        :type root: TreeNode
        :rtype: None Do not return anything, modify root in-place instead.
        """
        if root is None:
            return
        if root.left is None and root.right is None:
            return
        self.flatten(root.left)
        self.flatten(root.right)
        tmp = root.right
        root.right = root.left
        root.left = None
        while root.right:
            root = root.right
        root.right = tmp

# 115. Distinct Subsequences


class Solution(object):
    def numDistinct(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: int
        """
        len_s = len(s)
        len_t = len(t)
        res = [[0 for i in range(len_t + 1)] for j in range(len_s + 1)]
        for i in range(len_s + 1):
            res[i][0] = 1
        for row in range(1, len_s + 1, 1):
            for col in range(1, len_t + 1, 1):
                if s[row - 1] == t[col - 1]:
                    res[row][col] = res[row - 1][col] + res[row - 1][col - 1]
                else:
                    res[row][col] = res[row - 1][col]
        return res[len_s][len_t]

# 116. Populating Next Right Pointers in Each Node


class Node(object):
    def __init__(self, val, left, right, next):
        self.val = val
        self.left = left
        self.right = right
        self.next = next


class Solution(object):
    def connect(self, root):
        """
        :type root: Node
        :rtype: Node
        """
        if root and root.left:
            root.left.next = root.right
            if root.next:
                root.right.next = root.next.left
            self.connect(root.left)
            self.connect(root.right)
        return root


class Solution(object):
    def connect(self, root):
        """
        :type root: Node
        :rtype: Node
        """
        if root:
            tmp_iteration = root
            tmp = None
            next_layer_begin = None
            while tmp_iteration:
                if tmp_iteration.left:
                    if tmp:
                        tmp.next = tmp_iteration.left
                    tmp = tmp_iteration.left
                    if not next_layer_begin:
                        next_layer_begin = tmp_iteration.left
                if tmp_iteration.right:
                    if tmp:
                        tmp.next = tmp_iteration.right
                    tmp = tmp_iteration.right
                    if not next_layer_begin:
                        next_layer_begin = tmp_iteration.right
                tmp_iteration = tmp_iteration.next
            self.connect(next_layer_begin)
        return root

# 118. Pascal's Triangle


class Solution(object):
    def generate(self, numRows):
        """
        :type numRows: int
        :rtype: List[List[int]]
        """
        if numRows == 0:
            return []
        if numRows == 1:
            return [[1]]
        if numRows == 2:
            return [[1], [1, 1]]
        res = [[1], [1, 1]]
        for i in range(2, numRows):
            tmp = []
            tmp.append(1)
            for j in range(1, i):
                tmp.append(res[i - 1][j - 1] + res[i - 1][j])
            tmp.append(1)
            res.append(tmp)
        return res

# 119. Pascal's Triangle II


class Solution(object):
    def getRow(self, rowIndex):
        """
        :type rowIndex: int
        :rtype: List[int]
        """
        if rowIndex == 0:
            return [1]
        if rowIndex == 1:
            return [1, 1]
        res = [1, 1]
        for i in range(2, rowIndex + 1):
            tmp = []
            tmp.append(1)
            for j in range(1, i):
                tmp.append(res[j - 1] + res[j])
            tmp.append(1)
            res = tmp[:]
        return res


# 120. Triangle
class Solution(object):
    def minimumTotal(self, triangle):
        """
        :type triangle: List[List[int]]
        :rtype: int
        """
        len_t = len(triangle)
        if len_t == 0:
            return 0
        res = [0 for i in range(len(triangle))]
        res[0] = triangle[0][0]
        for row in range(1, len_t, 1):
            for col in range(len(triangle[row]) - 1, -1, -1):
                if col == len(triangle[row]) - 1:
                    res[col] = triangle[row][col] + res[col - 1]
                elif col == 0:
                    res[col] = triangle[row][col] + res[col]
                else:
                    res[col] = min(res[col],res[col - 1]) + triangle[row][col]
        return min(res)


if __name__ == '__main__':
    solu = Solution()
    test = [
        [2],
        [3, 4],
        [6, 5, 7],
        [4, 1, 8, 3]
    ]
    res = solu.minimumTotal(test)
    print res
