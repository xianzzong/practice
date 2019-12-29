# -*- coding: utf-8 -*
# 283. Move Zeroes


from heapq import *


class Solution(object):
    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        zero_index = 0
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[i], nums[zero_index] = nums[zero_index], nums[i]
                zero_index += 1
        return


# res = solu.moveZeroes([0, 1, 0, 3, 12])
# 282. Expression Add Operators


class Solution(object):
    def addOperators(self, num, target):
        """
        :type num: str
        :type target: int
        :rtype: List[str]
        """
        def isLeadingZeros(num):
            if num.startswith("00"):
                return True
            if int(num) and num.startswith("0"):
                return True
            return False

        def solve(num, target, mulExpr="", mulval=1):
            res = []
            if isLeadingZeros(num):
                pass
            elif int(num) * mulval == target:
                res.append(num + mulExpr)
            for x in range(len(num) - 1):
                left_num, right_num = num[:x + 1], num[x + 1:]
                if isLeadingZeros(right_num):
                    continue
                right = right_num + mulExpr
                right_value = int(right_num) * mulval
                # op="+"
                for left in solve(left_num, target - right_value):
                    res.append(left + "+" + right)
                # op="-"
                for left in solve(left_num, target + right_value):
                    res.append(left + "-" + right)
                # op="*"
                for left in solve(left_num, target, "*" + right, right_value):
                    res.append(left)
            return res
        if not num:
            return []
        return solve(num, target)


# res = solu.addOperators("105", 5)
# 284. Peeking Iterator
class PeekingIterator(object):
    def __init__(self, iterator):
        """
        Initialize your data structure here.
        :type iterator: Iterator
        """
        self.interator = Iterator()
        self.num = None

    def peek(self):
        """
        Returns the next element in the iteration without advancing the iterator.
        :rtype: int
        """
        if self.num is None:
            self.num = self.interator.next()
        return self.num

    def next(self):
        """
        :rtype: int
        """
        if self.num is None:
            return self.interator.next()
        else:
            tmp = self.num
            self.num = None
            return tmp

    def hasNext(self):
        """
        :rtype: bool
        """
        if self.num is None:
            return self.interator.hasNext()
        else:
            return True


# 287. Find the Duplicate Number
class Solution(object):
    def findDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        slow = nums[0]
        fast = nums[nums[0]]
        while slow != fast:
            fast = nums[nums[fast]]
            slow = nums[slow]
        fast = 0
        while slow != fast:
            fast = nums[fast]
            slow = nums[slow]
        return fast


# res = solu.findDuplicate([1, 3, 4, 2, 2])

# 289. Game of Life
class Solution(object):
    def gameOfLife(self, board):
        """
        :type board: List[List[int]]
        :rtype: None Do not return anything, modify board in-place instead.
        """
        # 0: 0 -> 0
        # 1: 1 -> 1
        # 2: 1 -> 0
        # 3: 0 -> 1

        def solve(x, y, row, col):
            dx = [1, 1, 1, 0, 0, -1, -1, -1]
            dy = [1, 0, -1, 1, -1, 1, 0, -1]
            cnt = 0
            for k in range(8):
                nx = x + dx[k]
                ny = y + dy[k]
                if nx < 0 or ny < 0 or nx >= row or ny >= col:
                    continue
                if board[nx][ny] == 1 or board[nx][ny] == 2:
                    cnt += 1
            return cnt
        row = len(board)
        if row == 0:
            return
        col = len(board[0])
        if col == 0:
            return
        for i in range(row):
            for j in range(col):
                cnt = solve(i, j, row, col)
                if board[i][j]:
                    if cnt < 2 or cnt > 3:
                        board[i][j] = 2
                else:
                    if cnt == 3:
                        board[i][j] = 3
        for i in range(row):
            for j in range(col):
                board[i][j] = board[i][j] & 1
        return
    # test = [
    #     [0, 1, 0],
    #     [0, 0, 1],
    #     [1, 1, 1],
    #     [0, 0, 0]
    # ]
    # res = solu.gameOfLife(test)

# 290. Word Pattern


class Solution(object):
    def wordPattern(self, pattern, str):
        """
        :type pattern: str
        :type str: str
        :rtype: bool
        """
        res_dict = {}
        str_data = str.split()
        len_s = len(str_data)
        len_p = len(pattern)
        if len_p != len_s:
            return False
        for i in range(len_p):
            if pattern[i] in res_dict:
                if str_data[i] != res_dict[pattern[i]]:
                    return False
            else:
                if str_data[i] in res_dict.values():
                    return False
                res_dict[pattern[i]] = str_data[i]
        return True


# 292. Nim Game
class Solution(object):
    def canWinNim(self, n):
        """
        :type n: int
        :rtype: bool
        """
        if n % 4 == 0:
            return False
        else:
            return True


# 295. Find Median from Data Stream
class MedianFinder(object):
    # 大顶堆中存储的元素 均不大于 小顶堆中的元素
    # MaxHeap.size() == MinHeap.size()，或者 MaxHeap.size() == MinHeap.size() + 1
    # 则有：

    # 当MaxHeap.size() == MinHeap.size() + 1时，中位数就是MaxHeap的堆顶元素
    # 当MaxHeap.size() == MinHeap.size()时，中位数就是MaxHeap堆顶元素与MinHeap堆顶元素的均值
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.minheap = []
        self.maxheap = []

    def addNum(self, num):
        """
        :type num: int
        :rtype: None
        """
        heappush(self.maxheap, -num)
        mintop = None
        maxtop = None
        if len(self.minheap):
            mintop = self.minheap[0]
        if len(self.maxheap):
            maxtop = self.maxheap[0]
        if mintop < -maxtop or len(self.minheap) + 1 < len(self.maxheap):
            heappush(self.minheap, -heappop(self.maxheap))
        if len(self.maxheap) < len(self.minheap):
            heappush(self.maxheap, -heappop(self.minheap))

    def findMedian(self):
        """
        :rtype: float
        """
        if len(self.minheap) < len(self.maxheap):
            return - 1.0 * self.maxheap[0]
        else:
            return (self.minheap[0]-self.maxheap[0])/2.0


# 297. Serialize and Deserialize Binary Tree
# Definition for a binary tree node.


class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.

        :type root: TreeNode
        :rtype: str
        """
        def doit(node):
            if node is None:
                vals.append("*")
            else:
                vals.append(str(node.val))
                doit(node.left)
                doit(node.right)
        vals = []
        doit(root)
        return " ".join(vals)

    def deserialize(self, data):
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: TreeNode
        """
        def doit():
            val = next(vals)
            if val == "*":
                return None
            else:
                node = TreeNode(int(val))
                node.left = doit()
                node.right = doit()
            return node
        vals = iter(data.split())
        res = doit()
        return res


# 299. Bulls and Cows
class Solution(object):
    def getHint(self, secret, guess):
        """
        :type secret: str
        :type guess: str
        :rtype: str
        """
        B_dict = {}
        B_num = 0
        A_num = 0
        for item in secret:
            if item in B_dict:
                B_dict[item] += 1
            else:
                B_dict[item] = 1
        for i in range(len(guess)):
            if guess[i] in B_dict:
                if B_dict[guess[i]] != 0:
                    B_dict[guess[i]] -= 1
                    B_num += 1
                if guess[i] == secret[i]:
                    A_num += 1
                    B_num -= 1
        res = "{}A{}B".format(A_num, B_num)
        return res


# res = solu.getHint("1807", "7810")

# 300. Longest Increasing Subsequence
class Solution(object):
    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        length = len(nums)
        if length == 0:
            return 0
        res = [1 for i in range(length)]
        for i in range(1, length, 1):
            for j in range(0, i, 1):
                if nums[i] > nums[j]:
                    res[i] = max(res[i], res[j]+1)
        return max(res)


if __name__ == "__main__":
    solu = Solution()
    res = solu.lengthOfLIS([10, 9, 2, 5, 3, 7, 101, 18])
    print(res)
