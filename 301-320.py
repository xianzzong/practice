# -*- coding: utf-8 -*
# 301. Remove Invalid Parentheses


class Solution(object):
    def removeInvalidParentheses(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        if not s:
            return [""]
        s_item = [s]
        res = []
        all_s = set([s])
        found = False
        while s_item:
            curr = s_item.pop(0)
            if self.isValidParentheses(curr):
                found = True
                res.append(curr)
            elif not found:
                for i in range(len(curr)):
                    if curr[i] == "(" or curr[i] == ")":
                        tmp = curr[:i] + curr[i + 1:]
                        if tmp not in all_s:
                            s_item.append(tmp)
                            all_s.add(tmp)
        return res

    def isValidParentheses(self, s):
        cnt = 0
        for item in s:
            if item == "(":
                cnt += 1
            elif item == ")":
                cnt -= 1
            if cnt < 0:
                return False
        return cnt == 0


# res = solu.removeInvalidParentheses("()())()")

# 303. Range Sum Query - Immutable
class NumArray(object):

    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        self.nums = nums

    def sumRange(self, i, j):
        """
        :type i: int
        :type j: int
        :rtype: int
        """
        res = sum(self.nums[i:j+1])
        return res


# 304. Range Sum Query 2D - Immutable
class NumMatrix(object):

    def __init__(self, matrix):
        """
        :type matrix: List[List[int]]
        """
        if not matrix or not matrix[0]:
            row, col = 0, 0
        else:
            row, col = len(matrix), len(matrix[0])
        self.sum_matrix = [[0] * (col + 1) for i in range(row + 1)]
        for i in range(row):
            for j in range(col):
                self.sum_matrix[i+1][j+1] = self.sum_matrix[i+1][j] + \
                    self.sum_matrix[i][j+1]-self.sum_matrix[i][j]+matrix[i][j]

    def sumRegion(self, row1, col1, row2, col2):
        """
        :type row1: int
        :type col1: int
        :type row2: int
        :type col2: int
        :rtype: int
        """
        res = self.sum_matrix[row2+1][col2+1] - self.sum_matrix[row2 +
                                                                1][col1]-self.sum_matrix[row1][col2+1]+self.sum_matrix[row1][col1]
        return res


# 306. Additive Number
class Solution(object):
    def isAdditiveNumber(self, num):
        """
        :type num: str
        :rtype: bool
        """
        return self.solve(num, [])

    def solve(self, num_str, path):
        if len(path) >= 3 and path[-1] != path[-2] + path[-3]:
            return False
        if not num_str and len(path) >= 3:
            return True
        for i in range(len(num_str)):
            curr = num_str[:i + 1]
            if curr[0] == "0" and len(curr) != 1:
                continue
            if self.solve(num_str[i + 1:], path + [int(curr)]):
                return True
        return False


# test="112358"

# 307. Range Sum Query - Mutable
class NumArray(object):

    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        self.nums = nums

    def update(self, i, val):
        """
        :type i: int
        :type val: int
        :rtype: None
        """
        self.nums[i] = val

    def sumRange(self, i, j):
        """
        :type i: int
        :type j: int
        :rtype: int
        """
        res = sum(self.nums[i:j+1])
        return res


# 309. Best Time to Buy and Sell Stock with Cooldown
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        # sells[i]表示在第i天卖出股票所能获得的最大累积收益
        # buys[i]表示在第i天买入股票所能获得的最大累积收益
        # 初始化令sells[0] = 0，buys[0] = -prices[0]
        # sells[i] = max(buys[i - 1] + prices[i], sells[i - 1] + delta)
        # buys[i] = max(sells[i - 2] - prices[i], buys[i - 1] - delta)
        # 第i天卖出的最大累积收益 = max(第i-1天买入~第i天卖出的最大累积收益, 第i-1天卖出后反悔~改为第i天卖出的最大累积收益)
        # 第i天买入的最大累积收益 = max(第i-2天卖出~第i天买入的最大累积收益, 第i-1天买入后反悔~改为第i天买入的最大累积收益)
        # 第i-1天卖出后反悔，改为第i天卖出 等价于 第i-1天持有股票，第i天再卖出
        # 第i-1天买入后反悔，改为第i天买入 等价于 第i-1天没有股票，第i天再买入
        length = len(prices)
        if length == 0:
            return 0
        buys = [None] * length
        sells = [None] * length
        sells[0] = 0
        buys[0] = -prices[0]
        for i in range(1, length, 1):
            delta = prices[i] - prices[i - 1]
            sells[i] = max(buys[i - 1] + prices[i], sells[i - 1] + delta)
            if i > 1:
                buys[i] = max(sells[i - 2] - prices[i], buys[i - 1] - delta)
            else:
                buys[i] = buys[i - 1] - delta
        return max(sells)


# 310. Minimum Height Trees
class Solution(object):
    def findMinHeightTrees(self, n, edges):
        """
        :type n: int
        :type edges: List[List[int]]
        :rtype: List[int]
        """
        children = [set() for i in range(n)]
        for s, t in edges:
            children[s].add(t)
            children[t].add(s)
        leaves = [x for x in range(n) if len(children[x]) <= 1]
        while n > 2:
            n = n - len(leaves)
            new_leaves = []
            for x in leaves:
                for y in children[x]:
                    children[y].remove(x)
                    if len(children[y]) == 1:
                        new_leaves.append(y)
            leaves = new_leaves
        return leaves
    # n = 4
    # edges = [[1, 0], [1, 2], [1, 3]]

# 312. Burst Balloons


class Solution(object):
    def maxCoins(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # dp[i][j]为打破的气球为i~j之间。
        # 那么介于i, j之间的x，有： dp[i][j] = max(dp[i][j], dp[i][x – 1] + nums[i – 1] * nums[x] * nums[j + 1] + dp[x + 1][j])
        length = len(nums)
        nums = [1] + nums + [1]
        res = [[0 for i in range(length + 2)] for j in range(length + 2)]

        def solve(i, j):
            if res[i][j] > 0:
                return res[i][j]
            for x in range(i, j + 1, 1):
                res[i][j] = max(res[i][j], solve(i, x-1)+nums[i-1]
                                * nums[x] * nums[j + 1] + solve(x + 1, j))
            return res[i][j]
        return solve(1, length)
# test = [3, 1, 5, 8]


# 313. Super Ugly Number
class Solution(object):
    def nthSuperUglyNumber(self, n, primes):
        """
        :type n: int
        :type primes: List[int]
        :rtype: int
        """
        result = [1]
        length = len(primes)
        primes_mul = [0 for i in range(length)]
        while True:
            tmp = []
            for i in range(length):
                tmp.append(result[primes_mul[i]] * primes[i])
            min_tmp = min(tmp)
            result.append(min_tmp)
            if len(result) >= n:
                break
            for i in range(length):
                if tmp[i] == min_tmp:
                    primes_mul[i] += 1
        return result[n-1]


# res = solu.nthSuperUglyNumber(12, [2, 7, 13, 19])

# 315. Count of Smaller Numbers After Self


class FenwickTree(object):
    def __init__(self, n):
        self.sum_array = [0] * (n + 1)
        self.n = n

    def lowbit(self, x):
        return x & -x

    def add(self, x, val):
        while x <= self.n:
            self.sum_array[x] += val
            x += self.lowbit(x)

    def sum(self, x):
        res = 0
        while x > 0:
            res += self.sum_array[x]
            x -= self.lowbit(x)
        return res


class Solution(object):
    def countSmaller(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        dic = {}
        for i, num in enumerate(sorted(list(set(nums)))):
            dic[num] = i + 1
        tree = FenwickTree(len(nums))
        ans = [0] * len(nums)
        for i in xrange(len(nums) - 1, -1, -1):
            ans[i] = tree.sum(dic[nums[i]] - 1)
            tree.add(dic[nums[i]], 1)
        return ans


# 316. Remove Duplicate Letters
class Solution(object):
    def removeDuplicateLetters(self, s):
        """
        :type s: str
        :rtype: str
        """
        # 解法是真的骚。。。看了半天才看明白。。
        visit = [False] * 26
        cnt = [0] * 26
        res = []
        for item in s:
            cnt[ord(item) - 97] += 1
        for item in s:
            index = ord(item) - 97
            cnt[index] -= 1
            if visit[index]:
                continue
            while res and res[-1] > item and cnt[ord(res[-1]) - 97] > 0:
                visit[ord(res.pop()) - 97] = False
            res.append(item)
            visit[index] = True
        return "".join(res)


# res = solu.removeDuplicateLetters("cbacdcbc")

# 318. Maximum Product of Word Lengths
class Solution(object):
    def maxProduct(self, words):
        """
        :type words: List[str]
        :rtype: int
        """
        # 空间换时间
        length = len(words)
        elements = [0] * length
        for i, s in enumerate(words):
            for c in s:
                elements[i] |= 1 << (ord(c) - 97)
        res = 0
        for i in range(length):
            for j in range(i + 1, length):
                if elements[i] & elements[j] == 0:
                    res = max(res, len(words[i]*len(words[j])))
        return res
    # res = solu.maxProduct(
    #     ["abcw", "baz", "foo", "bar", "xtfn", "abcdef"])


# 319. Bulb Switcher
class Solution(object):
    def bulbSwitch(self, n):
        """
        :type n: int
        :rtype: int
        """
        return int(math.sqrt(n))


if __name__ == "__main__":
    solu = Solution()
    res = solu.maxProduct(
        ["abcw", "baz", "foo", "bar", "xtfn", "abcdef"])
    print(res)
