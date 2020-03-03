# -*- coding: utf-8 -*
# 363. Max Sum of Rectangle No Larger Than K
# 这个好难啊，网上学来的
# First just create a prefix sum of each row
# Next we are going to do sliding window by starting off with rectangles of col length 1, then 2, then 3, then 4, and sliding those to every single possible spot. We just take each x1 and x2 and move them around all possible y spots.
# This is probably the most difficult to understand part. There's a special technique to drop the complexity from n^2 to nlogn. We can run an accumulator, and remember the past sums we found in a balanced tree, and every time we are on the next step, we can subtract from our current accumulated sum any old accumulated sum to get the true sum of a particular sub-rectangle. This lets us avoid manually checking each rectangle formed when sliding down the rows, and just let's us do a linear sweep, and with each operation, doing a log(n) search to find the closest fitting element within our RB tree. Each time we find a possible solution, just check if the maximum is greater than the current.


class Solution:
    def maxSumSubmatrix(self, matrix, k: int) -> int:
        m = len(matrix)
        n = len(matrix[0])
        maxSum = -sys.maxsize

        # prefix sum
        for i in range(m):
            for j in range(1, n):
                matrix[i][j] += matrix[i][j-1]

        for x1 in range(n):
            for x2 in range(x1, n):
                rectAcc = 0
                seen = SortedList()
                seen.add(0)
                for y in range(m):
                    rectAcc += matrix[y][x2] - \
                        (matrix[y][x1 - 1] if x1 > 0 else 0)
                    currTargetLoc = seen.bisect_left(rectAcc - k)

                    if(currTargetLoc < len(seen)):
                        maxSum = max(maxSum, rectAcc - seen[currTargetLoc])

                    seen.add(rectAcc)

        return maxSum


# 365. Water and Jug Problem
class Solution:
    def canMeasureWater(self, x: int, y: int, z: int) -> bool:
        def gcd(a, b):
            if b == 0:
                return a
            else:
                return gcd(b, a % b)
        if x + y == z:
            return True
        else:
            data = gcd(x, y)

            if (x + y) > z and z % gcd(x, y) == 0:
                return True
        return False


# 367. Valid Perfect Square
class Solution:
    def isPerfectSquare(self, num: int) -> bool:
        left = 1
        right = int(num / 2) + 1
        while left <= right:
            middle = left + int((right - left) / 2)
            mul = middle ** 2
            if mul == num:
                return True
            elif mul > num:
                right = middle - 1
            else:
                left = middle + 1
        return False


# 368. Largest Divisible Subset
class Solution:
    def largestDivisibleSubset(self, nums):
        if not nums:
            return []
        nums.sort()
        length = len(nums)
        dp = [1] * length
        index = [-1] * length
        max_dp = 1
        max_index = 0
        for i in range(length):
            for j in range(i - 1, -1, -1):
                if nums[i] % nums[j] == 0 and dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
                    index[i] = j
            if max_dp < dp[i]:
                max_dp = dp[i]
                max_index = i
        result = []
        while max_index != -1:
            result.append(nums[max_index])
            max_index = index[max_index]
        return result


# 371. Sum of Two Integers
class Solution:
    def getSum(self, a: int, b: int) -> int:
        while True:
            tmp = (a & b) << 1
            a = a ^ b
            b = tmp
            if b == 0:
                break
        return a

    def getSum(self, a: int, b: int) -> int:
        MOD = 0xFFFFFFFF
        MAX_INT = 0x7FFFFFFF
        while b != 0:
            a, b = (a ^ b) & MOD, ((a & b) << 1) & MOD
        return a if a <= MAX_INT else ~(a & MAX_INT) ^ MAX_INT


# 372. Super Pow
class Solution:
    def superPow(self, a, b):
        b = map(str, b)
        b = ''.join(b)
        result = pow(a, int(b), 1337)
        return result


# res = solu.superPow(2, [1, 0])/

# 373. Find K Pairs with Smallest Sums
class Solution:
    def kSmallestPairs(self, nums1, nums2, k):
        import heapq
        if not nums1 or not nums2:
            return []
        result = []
        M = len(nums1)
        N = len(nums2)
        visited = set()
        heap = []
        heapq.heappush(heap, (nums1[0] + nums2[0], 0, 0))
        visited.add((0, 0))
        while heap and len(result) < k:
            _, i, j = heapq.heappop(heap)
            result.append([nums1[i], nums2[j]])
            if i < M-1 and (i + 1, j) not in visited:
                heapq.heappush(heap, (nums1[i + 1] + nums2[j], i + 1, j))
                visited.add((i + 1, j))
            if j < N - 1 and (i, j + 1) not in visited:
                heapq.heappush(heap, (nums1[i] + nums2[j + 1], i, j + 1))
                visited.add((i, j + 1))
        return result


# 374. Guess Number Higher or Lower
class Solution:
    def guessNumber(self, n: int) -> int:
        L, R = 1, n
        while L <= R:
            mid = L + ((R - L) >> 1)
            res = guess(mid)
            if res == 0:
                return mid
            elif res == 1:
                L = mid + 1
            else:
                R = mid - 1
        return L


# 375. Guess Number Higher or Lower II
class Solution:
    def getMoneyAmount(self, n: int) -> int:
        dp = [[0] * (n + 1) for _ in range(n)]
        res = self.solve(dp, 1, n)
        return res

    def solve(self, dp, left, right):
        if left >= right:
            return 0
        if dp[left][right]:
            return dp[left][right]
        # dp[left][right] = min(i + max(self.solve(dp, left, i - 1),
        #                        self.solve(dp, i + 1, right)) for i in range(left, right + 1))
        for i in range(left, right + 1, 1):
            tmp = i + max(self.solve(dp, left, i - 1),
                          self.solve(dp, i + 1, right))
            if dp[left][right] == 0:
                dp[left][right] = tmp
            else:
                dp[left][right] = min(dp[left][right], tmp)
        return dp[left][right]


# 376. Wiggle Subsequence
class Solution:
    def wiggleMaxLength(self, nums):
        if not nums:
            return 0
        length = len(nums)
        dp = [1] * length
        sign = [0] * length
        for i in range(1, length, 1):
            for j in range(i - 1, -1, -1):
                if dp[j] + 1 > dp[i] and (
                    sign[j] > 0 and nums[j] > nums[i] or
                    sign[j] < 0 and nums[j] < nums[i] or
                    sign[j] == 0 and nums[j] != nums[i]
                ):
                    sign[i] = 1 if nums[i] > nums[j] else - 1
                    dp[i] = dp[j] + 1
        return max(dp)


# res = solu.wiggleMaxLength([1, 7, 4, 9, 2, 5])

# 377. Combination Sum IV
class Solution:
    def combinationSum4(self, nums, target):
        res = [1] + [0] * target
        for i in range(1, target + 1, 1):
            for item in nums:
                if i >= item:
                    res[i] += res[i - item]
        return res[target]


# 378. Kth Smallest Element in a Sorted Matrix·
class Solution(object):
    def kthSmallest(self, matrix, k):
        """
        :type matrix: List[List[int]]
        :type k: int
        :rtype: int
        """
        n = len(matrix)
        L, R = matrix[0][0], matrix[n - 1][n - 1]
        while L < R:
            mid = L + ((R - L) >> 1)
            temp = self.search_lower_than_mid(matrix, n, mid)
            if temp < k:
                L = mid + 1
            else:
                R = mid
        return L

    def search_lower_than_mid(self, matrix, n, x):
        i, j = n - 1, 0
        cnt = 0
        while i >= 0 and j < n:
            if matrix[i][j] <= x:
                j += 1
                cnt += i + 1
            else:
                i -= 1
        return cnt


# 380. Insert Delete GetRandom O(1)
class RandomizedSet:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.result = []
        self.index = {}

    def insert(self, val: int) -> bool:
        """
        Inserts a value to the set. Returns true if the set did not already contain the specified element.
        """
        if val in self.index:
            return False
        self.index[val] = len(self.result)
        self.result.append(val)
        return True

    def remove(self, val: int) -> bool:
        """
        Removes a value from the set. Returns true if the set contained the specified element.
        """
        if val not in self.index:
            return False
        index = self.index[val]
        last = self.result[-1]
        self.index[last] = index
        self.result[index] = last
        self.result.pop()
        del self.index[val]
        return True

    def getRandom(self) -> int:
        """
        Get a random element from the set.
        """
        import random
        index = random.randint(0, len(self.result) - 1)
        return self.result[index]


if __name__ == "__main__":
    solu = Solution()
    res = solu.wiggleMaxLength([1, 7, 4, 9, 2, 5])
    print(res)
