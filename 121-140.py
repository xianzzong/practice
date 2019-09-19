# 121. Best Time to Buy and Sell Stock
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        max_profit = 0
        len_p = len(prices)
        if len_p == 0:
            return max_profit
        lowest_price = prices[0]
        for i in range(1, len_p):
            max_profit = max(max_profit, prices[i] - lowest_price)
            lowest_price = min(lowest_price, prices[i])
        return max_profit

# 122. Best Time to Buy and Sell Stock II


class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        max_profit = 0
        len_p = len(prices)
        if len_p == 0:
            return max_profit
        for i in range(1, len_p):
            if prices[i] > prices[i - 1]:
                max_profit += prices[i] - prices[i - 1]
        return max_profit

# 123. Best Time to Buy and Sell Stock III


class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        len_p = len(prices)
        if len_p == 0:
            return 0
        res = 0
        tmp_left = [0 for i in range(len_p)]
        tmp_right = tmp_left[:]
        lowest = prices[0]
        for i in range(1, len_p, 1):
            lowest = min(lowest, prices[i])
            tmp_left[i] = max(tmp_left[i - 1], prices[i] - lowest)
        highest = prices[len_p - 1]
        tmp_right[len_p - 1] = 0
        for i in range(len_p - 2, -1, -1):
            highest = max(highest, prices[i])
            tmp_right[i] = max(tmp_right[i + 1], highest - prices[i])
        for i in range(len_p):
            res = max(res, tmp_left[i] + tmp_right[i])
        return res

# 124. Binary Tree Maximum Path Sum


class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution(object):
    def maxPathSum(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        def solve(node):
            if solve is None:
                return 0
            val_l = 0
            val_r = 0
            if node.left:
                val_l = max(solve(node.left), 0)
            if node.right:
                val_r = max(solve(node.right), 0)
            Solution.max = max(Solution.max, val_l + val_r + node.val)
            return max(val_l, val_r) + node.val
        if root is None:
            return 0
        Solution.max = root.val
        solve(root)
        return Solution.max

# 125. Valid Palindrome


class Solution(object):
    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        left = 0
        right = len(s) - 1
        if right <= 0:
            return True
        while left < right:
            if s[left].isalnum() and s[right].isalnum():
                if s[left].lower() == s[right].lower():
                    left += 1
                    right -= 1
                else:
                    return False
            elif s[left].isalnum():
                right -= 1
            elif s[right].isalnum():
                left += 1
            else:
                left += 1
                right -= 1
        return True

# 126. Word Ladder II


class Solution(object):
    def findLadders(self, beginWord, endWord, wordList):
        """
        :type beginWord: str
        :type endWord: str
        :type wordList: List[str]
        :rtype: List[List[str]]
        """
        def gen_path(path, word):
            if word == beginWord:
                path.append(word)
                tmp = path[:]
                tmp.reverse()
                result.append(tmp)
                path.pop()
                return
            path.append(word)
            for item in transition[word]:
                gen_path(path, item)
            path.pop()

        result = []
        if endWord not in wordList:
            return result
        wordDict = set(wordList)
        transition = {}
        start = set([beginWord])
        end = set()
        for item in wordDict:
            transition[item] = []
        while True:
            for item in start:
                for i in range(len(beginWord)):
                    for val in "abcdefghijklmnopqrstuvwxyz":
                        if val != item[i]:
                            newword = item[:i] + val + item[i + 1:]
                            if newword in wordDict:
                                end.add(newword)
                                transition[newword].append(item)
            print transition
            wordDict -= end
            if len(end) == 0:
                return []
            if endWord in end:
                break
            start = end.copy()
            end.clear()
        gen_path([], endWord)
        return result

# 127. Word Ladder


class Solution(object):
    def ladderLength(self, beginWord, endWord, wordList):
        """
        :type beginWord: str
        :type endWord: str
        :type wordList: List[str]
        :rtype: int
        """
        if endWord not in wordList:
            return 0
        wordDict = set(wordList)
        start = set([beginWord])
        end = set([endWord])
        length = 1
        while start:
            if start & end:
                return length
            tmp = set()
            for word in start:
                for i in range(len(beginWord)):
                    for value in "abcdefghijklmnopqrstuvwxyz":
                        tmp.add(word[:i] + value + word[i + 1:])
            start = wordDict & tmp
            print start, end
            length += 1
            if len(start) > len(end):
                start, end = end, start
            wordDict -= start
        return 0

# 128. Longest Consecutive Sequence


class Solution(object):
    def longestConsecutive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        len_n = len(nums)
        if len_n == 0:
            return 0
        max_len = 0
        nums_dict = {number: False for number in nums}
        for number in nums:
            if nums_dict[number] is True:
                continue
            else:
                nums_dict[number] = True
                num_left = 0
                num_right = 0
                tmp = number - 1
                while tmp in nums:
                    num_left += 1
                    tmp -= 1
                    nums_dict[tmp] = True
                tmp = number + 1
                while tmp in nums:
                    num_right += 1
                    tmp += 1
                    nums_dict[tmp] = True
                max_len = max(max_len, num_left + num_right + 1)
        return max_len

# 129. Sum Root to Leaf Numbers


class Solution(object):
    def sumNumbers(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        res = []

        def solve(node):
            if node.left is None and node.right is None:
                res.append(node.val)
            elif node.left is None:
                node.right.val += 10 * node.val
                solve(node.right)
            elif node.right is None:
                node.left.val += 10 * node.val
                solve(node.left)
            else:
                node.left.val += 10 * node.val
                solve(node.left)
                node.right.val += 10 * node.val
                solve(node.right)
        solve(root)
        print res
        return sum(res)


# 130. Surrounded Regions
class Solution(object):
    def solve(self, board):
        """
        :type board: List[List[str]]
        :rtype: None Do not return anything, modify board in-place instead.
        """
        def find_O(x, y):
            board[x][y] = "*"
            if 0 < x - 1 < row and board[x - 1][y] == 'O':
                find_O(x - 1, y)
            if 0 < x + 1 < row and board[x + 1][y] == 'O':
                find_O(x + 1, y)
            if 0 < y - 1 < col and board[x][y - 1] == 'O':
                find_O(x, y - 1)
            if 0 < y + 1 < col and board[x][y + 1] == 'O':
                find_O(x, y + 1)
            return
        row = len(board)
        if row == 0:
            return
        col = len(board[0])
        for i in range(row):
            if board[i][0] == "O":
                find_O(i, 0)
            if board[i][col - 1] == "O":
                find_O(i, col - 1)
        for i in range(1, col - 1, 1):
            if board[0][i] == "O":
                find_O(0, i)
            if board[row - 1][i] == "O":
                find_O(row - 1, i)

        for i in range(row):
            for j in range(col):
                if board[i][j] == "O":
                    board[i][j] = "X"
                if board[i][j] == "*":
                    board[i][j] = "O"
        return
    # test = [['x', 'x', 'x', 'x'],
    #         ['x', 'O', 'O', 'x'],
    #         ['x', 'x', 'O', 'x'],
    #         ['x', 'O', 'x', 'x']]
# 131. Palindrome Partitioning


class Solution(object):
    def partition(self, s):
        """
        :type s: str
        :rtype: List[List[str]]
        """
        def ispalindrome(sub_s):
            len_s = len(sub_s)
            for i in range(len_s):
                if sub_s[i] != sub_s[len_s - 1 - i]:
                    return False
            return True

        def solve(tmp_s, tmp_result):
            len_s = len(tmp_s)
            if len_s == 0:
                result.append(tmp_result)
            for i in range(1, len_s + 1, 1):
                if ispalindrome(tmp_s[:i]):
                    solve(tmp_s[i:], tmp_result + [tmp_s[:i]])
        result = []
        solve(s, [])
        return result
        test = 'aac'

# 132. Palindrome Partitioning II


class Solution(object):
    def minCut(self, s):
        """
        :type s: str
        :rtype: int
        """
        len_s = len(s)
        result = [i for i in range(len_s)]
        tmp = [[False for i in range(len_s)] for i in range(len_s)]
        tmp[0][0] = True
        j = 1
        while j < len_s:
            tmp[j][j] = True
            i = j - 1
            result[j] = min(result[j], result[j - 1] + 1)
            while i >= 0:
                if s[i] == s[j] and((j - i) < 2 or tmp[i + 1][j - 1]):
                    tmp[i][j] = True
                    if i == 0:
                        result[j] = 0
                    else:
                        result[j] = min(result[j], result[i - 1] + 1)
                i -= 1
            j += 1
        return result[len_s - 1]
        # test = "cdd"

# 133. Clone Graph


class Node(object):
    def __init__(self, val, neighbors):
        self.val = val
        self.neighbors = neighbors

# @param node, a undirected graph node
# @return a undirected graph node


class Solution(object):
    def cloneGraph(self, node):
        if node is None:
            return node
        tmp_g = {}

        def copygra(graph):
            if graph in tmp_g:
                return tmp_g[graph]
            result = Node(graph.val, [])
            tmp_g[graph] = result
            for nei in graph.neighbors:
                result.neighbors.append(copygra(nei))
            return result
        return copygra(node)

# 134. Gas Station


class Solution(object):
    def canCompleteCircuit(self, gas, cost):
        """
        :type gas: List[int]
        :type cost: List[int]
        :rtype: int
        """
        if sum(gas) < sum(cost):
            return -1
        len_gas = len(gas)
        begin = 0
        tmp_gas = 0
        i = 0
        while i < len_gas:
            tmp_gas += gas[i] - cost[i]
            if tmp_gas < 0:
                begin = i + 1
                tmp_gas = 0
            i += 1
        return begin
        # test = [[1,2,3,4,5],[3,4,5,1,2]]
# 135. Candy


class Solution(object):
    def candy(self, ratings):
        """
        :type ratings: List[int]
        :rtype: int
        """
        length = len(ratings)
        result = [1 for i in range(length)]
        for i in range(1, length, 1):
            if ratings[i] > ratings[i - 1]:
                result[i] = result[i - 1] + 1
        for i in range(length - 1, 0, -1):
            if ratings[i] < ratings[i - 1] and result[i] >= result[i - 1]:
                result[i - 1] = result[i] + 1
        print result
        return sum(result)
        # test = [1, 0, 2]
# 136. Single Number


class Solution(object):
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        res = 0
        for item in nums:
            res = res ^ item
        return res

# 137. Single Number II


class Solution(object):
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        one, two, three = 0, 0, 0
        for item in nums:
            two = two | (one & item)
            one = one ^ item
            three = ~(one & two)
            one = three & one
            two = three & two
        return one
        # test = [2, 2, 2, 34]

# 138. Copy List with Random Pointer
# Definition for a Node.


class Node(object):
    def __init__(self, val, next, random):
        self.val = val
        self.next = next
        self.random = random


class Solution(object):
    def copyRandomList(self, head):
        """
        :type head: Node
        :rtype: Node
        """
        if head == None:
            return head
        tmp = head
        while tmp:
            newnode = Node(tmp.val, None, None)
            newnode.next = tmp.next
            tmp.next = newnode
            tmp = tmp.next.next
        tmp = head
        while tmp:
            if tmp.random:
                tmp.next.random = tmp.random.next
            tmp = tmp.next.next
        NewNode = head.next
        pold = head
        pnew = NewNode
        while pnew.next:
            pold.next = pnew.next
            pold = pold.next
            pnew.next = pold.next
            pnew = pnew.next
        pold.next = None
        pnew.next = None
        return NewNode

# 139. Word Break


class Solution(object):
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """
        len_s = len(s)
        result = [False for i in range(len_s + 1)]
        result[0] = True
        for i in range(1, len_s + 1, 1):
            for j in range(i):
                if result[j] and s[j:i] in wordDict:
                    result[i] = True
                    break
        return result[len_s]

# 140. Word Break II


class Solution(object):
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: List[str]
        """
        def isBreak(s, wordDict):
            len_s = len(s)
            result = [False for i in range(len_s + 1)]
            result[0] = True
            for i in range(1, len_s + 1, 1):
                for j in range(i):
                    if result[j] and s[j:i] in wordDict:
                        result[i] = True
                        break
            return result[len_s]
        result = []
        if not isBreak(s, wordDict):
            return result
        len_s = len(s)
        if s in wordDict:
            result.append(s)
        for i in range(1, len_s + 1, 1):
            tmp = s[:i]
            if tmp in wordDict:
                if isBreak(s[i:], wordDict):
                    tmp_s = self.wordBreak(s[i:], wordDict)
                    for item in tmp_s:
                        result.append(tmp + " " + item)
        return result


if __name__ == '__main__':
    solu = Solution()
    s = "catsanddog"
    wordDict = ["cat", "cats", "and", "sand", "dog"]
    res = solu.wordBreak(s, wordDict)
    print res
