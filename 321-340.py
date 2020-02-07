# -*- coding: utf-8 -*
# 321. Create Maximum Number


class Solution(object):
    def maxNumber(self, nums1, nums2, k):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :type k: int
        :rtype: List[int]
        """
        # 不会做，往上找的答案T=T
        def get_max_sub_array(nums, k):
            res = []
            length = len(nums)
            for i in range(n):
                while res and len(res) + n - i > k and nums[i] > res[-1]:
                    res.pop()
                if len(res) < k:
                    res.append(nums[i])
            return res
        result = [0] * k
        for i in range(max(0, k - len(nums2)), min(k, len(nums1)) + 1):
            res1 = get_max_sub_array(nums1, i)
            res2 = get_max_sub_array(nums2, k - i)
            result = max(ans, [max(res1, res2).pop(0) for _ in xrange(k)])
        return result


# 322. Coin Change
class Solution(object):
    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        # dp，设result[i] 为兑换目标i最少的硬币数。
        # result[i+coins[j]] = min(result[i+coins[j]], result[i]+1)
        INF = 0x7ffffffe
        result = [INF] * (amount + 1)
        result[0] = 0
        for i in range(amount + 1):
            for coin in coins:
                if i + coin <= amount:
                    result[i + coin] = min(result[i + coin], result[i] + 1)
        if result[amount] == INF:
            result[amount] = -1
        return result[amount]


# res = solu.coinChange([1, 2, 5], 11)

# 324. Wiggle Sort II
class Solution(object):
    def wiggleSort(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        tmp = sorted(nums)
        left = (len(nums) + 1) >> 1
        right = len(nums)
        for i in range(len(nums)):
            if i & 1 == 0:
                left -= 1
                nums[i] = tmp[left]
            else:
                right -= 1
                nums[i] = tmp[right]
        return


# res = solu.wiggleSort([1, 5, 1, 1, 6, 4])
# 326. Power of Three
class Solution(object):
    def isPowerOfThree(self, n):
        """
        :type n: int
        :rtype: bool
        """
        if n <= 0:
            return False
        if n == 1:
            return True
        res = n / 3
        remain = n % 3
        while res > 1:
            if remain != 0:
                return False
            else:
                tmp = res / 3
                remain = res % 3
                res = tmp
        if res == 1 and remain == 0:
            return True
        else:
            return False


# res = solu.isPowerOfThree(27)
# 327. Count of Range Sum
# 这个题目感觉有点儿扯淡===

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
    def countRangeSum(self, nums, lower, upper):
        """
        :type nums: List[int]
        :type lower: int
        :type upper: int
        :rtype: int
        """
        if not nums:
            return 0
        sum_array = [upper, lower - 1]
        total = 0
        for num in nums:
            total += num
            sum_array += [total, total + lower - 1, total + upper]

        index = {}
        for i, x in enumerate(sorted(set(sum_array))):
            index[x] = i + 1

        tree = FenwickTree(len(index))
        ans = 0
        for i in xrange(len(nums) - 1, -1, -1):
            tree.add(index[total], 1)
            total -= nums[i]
            ans += tree.sum(index[upper + total]) - \
                tree.sum(index[lower + total - 1])
        return ans


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
    def countRangeSum(self, nums, lower, upper):
        """
        :type nums: List[int]
        :type lower: int
        :type upper: int
        :rtype: int
        """
        if not nums:
            return 0
        sum_array = [upper, lower - 1]
        total = 0
        for num in nums:
            total += num
            sum_array += [total, total + lower - 1, total + upper]

        index = {}
        for i, x in enumerate(sorted(set(sum_array))):
            index[x] = i + 1

        tree = FenwickTree(len(index))
        ans = 0
        for i in xrange(len(nums) - 1, -1, -1):
            tree.add(index[total], 1)
            total -= nums[i]
            ans += tree.sum(index[upper + total]) - \
                tree.sum(index[lower + total - 1])
        return ans


# 328. Odd Even Linked List
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution(object):
    def oddEvenList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head:
            return head
        p = head
        q = head
        while q:
            q = q.next
            if not q or not q.next:
                break
            next_p = p.next
            next_q = q.next
            q.next = next_q.next
            p.next, next_q.next = next_q, next_p
            p = p.next
        return head


# 329. Longest Increasing Path in a Matrix
class Solution(object):
    def longestIncreasingPath(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: int
        """
        if not matrix:
            return 0
        row = len(matrix)
        col = len(matrix[0])
        res = [[0 for i in range(col)] for j in range(row)]
        result = 0
        for i in range(row):
            for j in range(col):
                result = max(result, self.solve(i, j, row, col, matrix, res))
        return result

    def solve(self, x, y, row, col, matrix, res):
        if res[x][y]:
            return res[x][y]
        for dx, dy in ([(1, 0), (-1, 0), (0, 1), (0, -1)]):
            nx, ny = x + dx, y + dy
            if nx >= 0 and nx < row and ny >= 0 and ny < col and matrix[x][y] < matrix[nx][ny]:
                res[x][y] = max(res[x][y], self.solve(
                    nx, ny, row, col, matrix, res))
        res[x][y] += 1
        return res[x][y]


# 330. Patching Array
class Solution(object):
    def minPatches(self, nums, n):
        """
        :type nums: List[int]
        :type n: int
        :rtype: int
        """

        # 就是用known_sum表示已知的连续和为[1, known_sum)，有了这个表示那就简单了：

        # nums[i] <= known_sum，更新已知范围为：[1, known_sum + nums[i])
        # nums[i] > known_sum,  添加known_sum进数组才能达到最大的范围，所以已知范围更新为：[1, known_sum * 2)
        i = 0
        cnt = 0
        known_sum = 1
        while known_sum <= n:
            if i < len(nums) and nums[i] <= known_sum:
                known_sum += nums[i]
                i += 1
            else:
                known_sum <<= 1
                cnt += 1
        return cnt
        # nums = [1, 3]
        # n = 6
        # res = solu.minPatches(nums, n)

# 331. Verify Preorder Serialization of a Binary Tree


class Solution(object):
    def isValidSerialization(self, preorder):
        """
        :type preorder: str
        :rtype: bool
        """
        stack = []
        data = preorder.split(",")
        for item in data:
            stack.append(item)
            while len(stack) >= 3 and stack[-1] == "#" and stack[-2] == "#" and stack[-3] != "#":
                stack = stack[:-3] + ["#"]
        if len(stack) == 1 and stack[0] == "#":
            return True
        else:
            return False


# res = solu.isValidSerialization("9,3,4,#,#,1,#,#,2,#,6,#,#")
# 332. Reconstruct Itinerary
class Solution(object):
    def findItinerary(self, tickets):
        """
        :type tickets: List[List[str]]
        :rtype: List[str]
        """
        routes = {}
        for s, e in tickets:
            if s in routes:
                routes[s].append(e)
            else:
                routes[s] = [e]

        def solve(start):
            left, right = [], []
            if start in routes:
                for end in sorted(routes[start]):
                    if end not in routes[start]:
                        continue
                    routes[start].remove(end)
                    subroutes = solve(end)
                    if start in subroutes:
                        left += subroutes
                    else:
                        right += subroutes
            return [start] + left + right
        return solve("JFK")
# res = solu.findItinerary(
#     [["MUC", "LHR"], ["JFK", "MUC"], ["SFO", "SJC"], ["LHR", "SFO"]])


# 334. Increasing Triplet Subsequence
class Solution(object):
    def increasingTriplet(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        length = len(nums)
        if length < 3:
            return False
        minvalue = 0x7fffffff
        minvalue_1 = 0x7fffffff
        for num in nums:
            if num <= minvalue:
                minvalue = num
            elif num <= minvalue_1:
                minvalue_1 = num
            else:
                return True
        return False


# res = solu.increasingTriplet([1, 2, 3, 4, 5])

# 335. Self Crossing
class Solution(object):
    def isSelfCrossing(self, x):
        """
        :type x: List[int]
        :rtype: bool
        """
        n = len(x)
        if n < 4:
            return False
        t0, (t1, t2, t3) = 0, x[:3]
        increase = True if t1 < t3 else False
        for i in xrange(3, n):
            t4 = x[i]
            if increase and t2 >= t4:
                if t4 + t0 - t2 < 0 or i + 1 < n and x[i + 1] + t1 - t3 < 0:
                    increase = False
                elif i + 1 < n:
                    return True
            elif not increase and t2 <= t4:
                return True
            t0, t1, t2, t3 = t1, t2, t3, t4
        return False


# res = solu.increasingTriplet([1, 2, 3, 4, 5])

# 336. Palindrome Pairs
class Solution(object):
    def palindromePairs(self, words):
        """
        :type words: List[str]
        :rtype: List[List[int]]
        """
        tmp_words = {}
        for i, w in enumerate(words):
            tmp_words[w] = i
        res = set()
        for index, word in enumerate(words):
            if word and self.isPalindrome(word) and "" in tmp_words:
                index_2 = tmp_words[""]
                res.add((index, index_2))
                res.add((index_2, index))
            rword = word[::-1]
            if word and rword in tmp_words:
                index_2 = tmp_words[rword]
                if index != index_2:
                    res.add((index, index_2))
                    res.add((index_2, index))
            for i in range(1, len(word)):
                left = word[:i]
                right = word[i:]
                rleft = left[::-1]
                rright = right[::-1]
                if self.isPalindrome(left) and rright in tmp_words:
                    res.add((tmp_words[rright], index))
                if self.isPalindrome(right) and rleft in tmp_words:
                    res.add((index, tmp_words[rleft]))
        return list(res)

    def isPalindrome(self, word):
        length = len(word)
        for i in range(length / 2):
            if word[i] != word[length - i - 1]:
                return False
        return True


# test = ["a", "b", "c", "ab", "ac", "aa"]

# 337. House Robber III
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution(object):
    def rob(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        return self.solve(root)[0]

    def solve(self, root):
        if not root:
            return 0, 0
        rob_l, no_rob_l = self.solve(root.left)
        rob_r, no_rob_r = self.solve(root.right)
        return max(no_rob_l+no_rob_r+root.val, rob_l+rob_r), rob_l + rob_r


# 338. Counting Bits
class Solution(object):
    def countBits(self, num):
        """
        :type num: int
        :rtype: List[int]
        """
        result = [0] * (num + 1)
        before = 1
        power = 1
        for i in range(1, num + 1):
            if i == power:
                result[i] = 1
                before = 1
                power <<= 1
            else:
                result[i] = result[before] + 1
                before += 1
        return result


if __name__ == "__main__":
    solu = Solution()
    res = solu.countBits(12)
    print(res)
