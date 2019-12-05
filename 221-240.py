# -*- coding: utf-8 -*
# 221. Maximal Square


class Solution(object):
    def maximalSquare(self, matrix):
        """
        :type matrix: List[List[str]]
        :rtype: int
        """
        row = len(matrix)
        if row <= 0:
            return 0
        col = len(matrix[0])
        if col <= 0:
            return 0
        max_square = 0
        res = [[0 for i in range(col)] for j in range(row)]
        for i in range(row):
            if matrix[i][0] == "1":
                res[i][0] = 1
                max_square = 1
        for j in range(col):
            if matrix[0][j] == "1":
                res[0][j] = 1
                max_square = 1
        for i in range(1, row, 1):
            for j in range(1, col, 1):
                if matrix[i][j] == "1":
                    res[i][j] = min(res[i - 1][j - 1], res[i - 1]
                                    [j], res[i][j - 1]) + 1
                    max_square = max(max_square, res[i][j])
                else:
                    res[i][j] = 0
        return max_square**2
    # test=[["1", "0", "1", "0", "0"],
    #         ["1", "0", "1", "1", "1"],
    #         ["1", "1", "1", "1", "1"],
    #         ["1", "0", "0", "1", "0"]]
# 222. Count Complete Tree Nodes


class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution(object):
    def countNodes(self, root):
        '''
        假如左子树高度等于右子树高度，则右子树为完全二叉树，左子树为满二叉树。
        假如高度不等，则左字数为完全二叉树，右子树为满二叉树。
        求高度的时候只往左子树来找。
       '''
        """
        :type root: TreeNode
        :rtype: int
        """

        if not root:
            return 0
        node = 0
        left_height = self.get_height(root.left)
        right_height = self.get_height(root.right)
        if left_height == right_height:
            node = 2**left_height + self.countNodes(root.right)
        else:
            node = 2**right_height + self.countNodes(root.left)
        return node

    def get_height(self, root):
        height = 0
        while root:
            height += 1
            root = root.left
        return height


# 223. Rectangle Area
class Solution(object):
    def computeArea(self, A, B, C, D, E, F, G, H):
        """
        :type A: int
        :type B: int
        :type C: int
        :type D: int
        :type E: int
        :type F: int
        :type G: int
        :type H: int
        :rtype: int
        """
        def cal_area(row_left, col_left, row_right, col_right):
            width = row_right - row_left
            height = col_right - col_left
            if width <= 0 or height <= 0:
                return 0
            else:
                return width * height
        inter_down_row = max(A, E)
        inter_down_col = max(B, F)
        inter_up_row = min(C, G)
        inter_up_col = min(D, H)
        inter_area = cal_area(inter_down_row, inter_down_col,
                              inter_up_row, inter_up_col)
        area1 = cal_area(A, B, C, D)
        area2 = cal_area(E, F, G, H)
        return area1 - inter_area + area2

        # res = solu.computeArea(A=-3, B=0, C=3, D=4, E=0, F=-1, G=9, H=2)
# 224. Basic Calculator


class Solution(object):
    def calculate(self, s):
        """
        :type s: str
        :rtype: int
        """
        res = 0
        sign = 1
        num = 0
        tmp = []
        for item in s:
            if item.isdigit():
                num = 10 * num + int(item)
            elif item == "+" or item == "-":
                res = res + sign * num
                num = 0
                if item == "+":
                    sign = 1
                else:
                    sign = -1
            elif item == "(":
                tmp.append(res)
                tmp.append(sign)
                res = 0
                sign = 1
            elif item == ")":
                res = res + sign * num
                num = 0
                res = res * tmp.pop()
                res = res + tmp.pop()
        res = res + sign * num
        return res

# 225. Implement Stack using Queues


class MyStack(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.stack = []

    def push(self, x):
        """
        Push element x onto stack.
        :type x: int
        :rtype: None
        """
        self.stack.append(x)

    def pop(self):
        """
        Removes the element on top of the stack and returns that element.
        :rtype: int
        """
        value = self.stack[-1]
        self.stack.remove(value)
        return value

    def top(self):
        """
        Get the top element.
        :rtype: int
        """
        value = self.stack[-1]
        return value

    def empty(self):
        """
        Returns whether the stack is empty.
        :rtype: bool
        """
        return len(self.stack) == 0

# 226. Invert Binary Tree


class Solution(object):
    def invertTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        if root is None:
            return root
        tmp = root.right
        root.right = self.invertTree(root.left)
        root.left = self.invertTree(tmp)
        return root

# 227. Basic Calculator II


class Solution(object):
    def calculate(self, s):
        """
        :type s: str
        :rtype: int
        """
        pre_op = "+"
        res_list = []
        num = 0
        length = len(s)
        for i in range(length):
            if s[i].isdigit():
                num = num * 10 + int(s[i])
            if s[i] in "+-*/" or i == length - 1:
                if pre_op == "+":
                    res_list.append(num)
                elif pre_op == "-":
                    res_list.append(-num)
                elif pre_op == "*":
                    tmp = res_list.pop()
                    res_list.append(tmp * num)
                else:
                    tmp = res_list.pop()
                    if tmp < 0:
                        res_list.append(-(-tmp / num))
                    else:
                        res_list.append(tmp / num)
                num = 0
                pre_op = s[i]
        return sum(res_list)
        # res = solu.calculate("14-3/2")
# 228. Summary Ranges


class Solution(object):
    def summaryRanges(self, nums):
        """
        :type nums: List[int]
        :rtype: List[str]
        """
        start = 0
        length = 1
        res = []
        if len(nums) == 0:
            return res
        for i in range(1, len(nums), 1):
            if nums[i] - nums[i - 1] == 1:
                length += 1
            else:
                if length == 1:
                    res.append(str(nums[start]))
                else:
                    tmp = str(nums[start]) + "->" + \
                        str(nums[start + length - 1])
                    res.append(tmp)
                start = i
                length = 1
        if length == 1:
            res.append(str(nums[start]))
        else:
            tmp = str(nums[start]) + "->" + str(nums[start + length - 1])
            res.append(tmp)
        return res


# res = solu.summaryRanges( [0,2,3,4,6,8,9])
# 229. Majority Element II
class Solution(object):
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        res = []
        num_one = 0
        num_two = 0
        cnt_one = 0
        cnt_two = 0
        for num in nums:
            if num == num_one:
                cnt_one += 1
            elif num == num_two:
                cnt_two += 1
            elif cnt_one == 0:
                num_one = num
                cnt_one = 1
            elif cnt_two == 0:
                num_two = num
                cnt_two = 1
            else:
                cnt_one -= 1
                cnt_two -= 1
        cnt_one = 0
        cnt_two = 0
        for num in nums:
            if num == num_one:
                cnt_one += 1
            elif num == num_two:
                cnt_two += 1
        length = len(nums)
        if cnt_one > length / 3:
            res.append(num_one)
        if cnt_two > length / 3:
            res.append(num_two)
        return res
# res = solu.majorityElement([1, 1, 1, 3, 3, 2, 2, 2])
# 230. Kth Smallest Element in a BST


class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution(object):
    def kthSmallest(self, root, k):
        """
        :type root: TreeNode
        :type k: int
        :rtype: int
        """
        stack = []
        node = root
        while node:
            stack.append(node)
            node = node.left
        x = 1
        while stack and x <= k:
            node = stack.pop()
            x += 1
            right = node.right
            while right:
                stack.append(right)
                right = right.left
        return node.val

# 231. Power of Two


class Solution(object):
    def isPowerOfTwo(self, n):
        """
        :type n: int
        :rtype: bool
        """
        cnt = 0
        if n <= 0:
            return False
        while n > 0:
            cnt += n & 1
            n = n >> 1
            if cnt > 1:
                return False
        return True

        # res = solu.isPowerOfTwo(2)

# 232. Implement Queue using Stacks


class MyQueue(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.data = []

    def push(self, x):
        """
        Push element x to the back of queue.
        :type x: int
        :rtype: None
        """
        self.data.append(x)

    def pop(self):
        """
        Removes the element from in front of queue and returns that element.
        :rtype: int
        """
        value = self.data[0]
        self.data.remove(value)
        return value

    def peek(self):
        """
        Get the front element.
        :rtype: int
        """
        return self.data[0]

    def empty(self):
        """
        Returns whether the queue is empty.
        :rtype: bool
        """
        return len(self.data) == 0

# 233. Number of Digit One
# case 1: n=3141092, a= 31410, b=92. 计算百位上1的个数应该为 3141 *100 次.
# case 2: n=3141192, a= 31411, b=92. 计算百位上1的个数应该为 3141 *100 + (92+1) 次.
# case 3: n=3141592, a= 31415, b=92. 计算百位上1的个数应该为 (3141+1) *100 次.
# 以上三种情况可以用 一个公式概括:
#
# (a + 8) / 10 * m + (a % 10 == 1) * (b + 1);


class Solution(object):
    def countDigitOne(self, n):
        """
        :type n: int
        :rtype: int
        """
        res = 0
        m = 1
        while m <= n:
            a = n / m
            b = n % m
            print a, b
            res += (a + 8) / 10 * m
            if a % 10 == 1:
                res += b + 1
            m = 10 * m
        return res
        # res = solu.countDigitOne(20)
# 234. Palindrome Linked List


class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution(object):
    def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        if head is None or head.next is None:
            return False
        stack = []
        while head:
            stack.append(head.val)
            head = head.next
        length = len(stack)
        for i in range(0, length / 2, 1):
            if stack[i] != stack[length - 1 - i]:
                return False
        return True

# 235. Lowest Common Ancestor of a Binary Search Tree


class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        if root is None:
            return root
        if p.val > q.val:
            return self.lowestCommonAncestor(root, q, p)
        if root.val >= p.val and root.val <= q.val:
            return root
        if root.val < p.val:
            return self.lowestCommonAncestor(root.right, p, q)
        if root.val > q.val:
            return self.lowestCommonAncestor(root.left, p, q)

# 236. Lowest Common Ancestor of a Binary Tree


class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        if root is None or root == p or root == q:
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        if left and right:
            return root
        elif left:
            return left
        elif right:
            return right


class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution(object):
    def deleteNode(self, node):
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        node.val = node.next.val
        tmp = node.next
        node.next = tmp.next
        return

# 238. Product of Array Except Self


class Solution(object):
    def productExceptSelf(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        res = [1]
        length = len(nums)
        for i in range(1, length, 1):
            res.append(res[i - 1] * nums[i - 1])
        right = 1
        for i in range(length - 1, -1, -1):
            res[i] = res[i] * right
            right = right * nums[i]
        return

# 239. Sliding Window Maximum


class Solution(object):
    def maxSlidingWindow(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        import collections
        dp = collections.deque()
        res = []
        for i in range(len(nums)):
            while dp and nums[dp[-1]] <= nums[i]:
                dp.pop()
            dp.append(i)
            if dp[0] == i - k:
                dp.popleft()
            if i >= k - 1:
                res.append(nums[dp[0]])
        return res
        # res = solu.maxSlidingWindow([1, 3, -1, -3, 5, 3, 6, 7], 3)

# 240. Search a 2D Matrix II


class Solution(object):
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        row = len(matrix)
        if row <= 0:
            return False
        col = len(matrix[0])
        if col <= 0:
            return False
        for i in range(row):
            if matrix[i][0] > target:
                return False
            if matrix[i][col - 1] < target:
                continue
            left = 0
            right = col - 1
            while left <= right:
                mid = (left + right) / 2
                if matrix[i][mid] == target:
                    return True
                elif matrix[i][mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
        return False


if __name__ == "__main__":
    solu = Solution()
    test = [
        [1,   4,  7, 11, 15],
        [2,   5,  8, 12, 19],
        [3,   6,  9, 16, 22],
        [10, 13, 14, 17, 24],
        [18, 21, 23, 26, 30]
    ]
    res = solu.searchMatrix(test, 20)
    print (res)
