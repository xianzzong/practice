# 162. Find Peak Element
class Solution(object):
    def findPeakElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        length = len(nums)
        if length <= 1:
            return 0
        left = 0
        right = length - 1
        while left <= right:
            if left == right:
                return left
            elif left + 1 == right:
                if nums[left] < nums[right]:
                    return right
                else:
                    return left
            mid = (left + right) / 2
            if nums[mid] < nums[mid - 1]:
                right = mid - 1
            elif nums[mid] < nums[mid + 1]:
                left = mid + 1
            else:
                return mid
                # test= [3, 2, 1]
# 164. Maximum Gap


class Solution(object):
    def maximumGap(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        length = len(nums)
        if length <= 1:
            return 0
        max_value = max(nums)
        min_value = min(nums)
        bucket_range = max(1, (max_value - min_value - 1) / (length - 1) + 1)
        bucket_len = (max_value - min_value) / bucket_range + 1
        buckets = [None] * bucket_len
        for item in nums:
            bucket_idex = (item - min_value) / bucket_range
            bucket = buckets[bucket_idex]
            if bucket is None:
                bucket = {"min": item, "max": item}
                buckets[bucket_idex] = bucket
            else:
                bucket["min"] = min(item, bucket["min"])
                bucket["max"] = max(item, bucket["max"])
        maxgap = buckets[0]["max"] - buckets[0]["min"]
        i = 0
        while i < bucket_len:
            if buckets[i] is None:
                continue
            j = i + 1
            while j < bucket_len and buckets[j] is None:
                j += 1
            if j < bucket_len:
                maxgap = max(maxgap, buckets[j]["min"] - buckets[i]["max"])
            i = j
        return maxgap
        # test = [3, 6, 9, 1]
# 165. Compare Version Numbers


class Solution(object):
    def compareVersion(self, version1, version2):
        """
        :type version1: str
        :type version2: str
        :rtype: int
        """
        v1 = version1.split('.')
        v2 = version2.split('.')
        len_1 = len(v1)
        len_2 = len(v2)
        length = min(len_1, len_2)
        i = 0
        while i < length:
            tmp1 = int(v1[i])
            tmp2 = int(v2[i])
            if tmp1 == tmp2:
                i += 1
            elif tmp1 < tmp2:
                return -1
            else:
                return 1
        if i == len_1 and i != len_2:
            while i < len_2:
                if int(v2[i]) > 0:
                    return -1
                i += 1
        elif i != len_1 and i == len_2:
            while i < len_1:
                if int(v1[i]) > 0:
                    return 1
                i += 1
        return 0
# 166. Fraction to Recurring Decimal


class Solution(object):
    def fractionToDecimal(self, numerator, denominator):
        """
        :type numerator: int
        :type denominator: int
        :rtype: str
        """
        if (numerator >= 0 and denominator > 0) or (numerator <= 0 and denominator < 0):
            result = ''
        else:
            result = '-'
        numerator = abs(numerator)
        denominator = abs(denominator)
        tmp_quotient = numerator / denominator
        tmp_remain = numerator % denominator
        result += str(tmp_quotient)
        quotient = []
        remain = []
        remain_index = -1
        print result
        while tmp_remain:
            remain.append(str(tmp_remain))
            tmp = tmp_remain * 10
            tmp_quotient = tmp / denominator
            tmp_remain = tmp % denominator
            quotient.append(str(tmp_quotient))
            if str(tmp_remain) in remain:
                remain_index = remain.index(str(tmp_remain))
                break
        if len(quotient) and remain_index != -1:
            result = result + "." + \
                "".join(quotient[:remain_index]) + \
                "(" + "".join(quotient[remain_index:]) + ")"
        elif len(quotient) and remain_index == -1:
            result = result + "." + "".join(quotient)
        return result
        # test = [1, 2]

# 167. Two Sum II - Input array is sorted


class Solution(object):
    def twoSum(self, numbers, target):
        """
        :type numbers: List[int]
        :type target: int
        :rtype: List[int]
        """
        len_n = len(numbers)
        left = 0
        right = len_n - 1
        while left < right:
            if numbers[left] + numbers[right] > target:
                right -= 1
            elif numbers[left] + numbers[right] < target:
                left += 1
            else:
                break
        if left == right:
            return 0
        return [left + 1, right + 1]
        # res = solu.twoSum([2, 7, 11, 15], 9)

# 168. Excel Sheet Column Title


class Solution(object):
    def convertToTitle(self, n):
        """
        :type n: int
        :rtype: str
        """
        res = ""
        while n:
            res = chr(ord("A") + (n - 1) % 26) + res
            n = (n - 1) / 26
        return res

# 169. Majority Element


class Solution(object):
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        result = nums[0]
        result_num = 1
        length = len(nums)
        for i in range(1, length, 1):
            if result_num == 0:
                result = nums[i]
                result_num = 1
            else:
                if nums[i] == result:
                    result_num += 1
                else:
                    result_num -= 1
        return result
# res = solu.majorityElement([10,9,9,9,10])
# 171. Excel Sheet Column Number


class Solution(object):
    def titleToNumber(self, s):
        """
        :type s: str
        :rtype: int
        """
        result = 0
        for item in s:
            result = result * 26 + ord(item) - 64
        return result
        # res = solu.titleToNumber("ZY")
# 172. Factorial Trailing Zeroes


class Solution(object):
    def trailingZeroes(self, n):
        """
        :type n: int
        :rtype: int
        """
        res = 0
        while n >= 5:
            res += n / 5
            n = n / 5
        return res

# 173. Binary Search Tree Iterator


class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class BSTIterator(object):

    def __init__(self, root):
        """
        :type root: TreeNode
        """
        self.result = []
        self.putleft(root)

    def next(self):
        """
        @return the next smallest number
        :rtype: int
        """
        res = self.result.pop()
        self.putleft(res.right)
        return res.val

    def hasNext(self):
        """
        @return whether we have a next smallest number
        :rtype: bool
        """
        length = len(self.result)
        if length == 0:
            return False
        else:
            return True

    def putleft(self, node):
        while node:
            self.result.append(node)
            node = node.left

# 174. Dungeon Game


class Solution(object):
    def calculateMinimumHP(self, dungeon):
        """
        :type dungeon: List[List[int]]
        :rtype: int
        """
        row = len(dungeon)
        if row == 0:
            return 0
        col = len(dungeon[0])
        if col == 0:
            return 0
        result = [[0 for i in range(col)] for j in range(row)]
        result[row - 1][col - 1] = max(0, -dungeon[row - 1][col - 1]) + 1
        for row_index in range(row - 1, -1, -1):
            for col_index in range(col - 1, -1, -1):
                down = 0
                if row_index + 1 < row:
                    down = max(
                        1, result[row_index + 1][col_index] - dungeon[row_index][col_index])
                right = 0
                if col_index + 1 < col:
                    right = max(
                        1, result[row_index][col_index + 1] - dungeon[row_index][col_index])
                if down and right:
                    result[row_index][col_index] = min(down, right)
                elif down:
                    result[row_index][col_index] = down
                elif right:
                    result[row_index][col_index] = right
        return result[0][0]
        # test = [[-2, -3, 3], [-5, -10, 1], [10, 30, -5]]

# 175. Combine Two Tables
# Write your MySQL query statement below
# Select Person.FirstName, Person.LastName, Address.City, Address.State
# from Person Left Join Address On Person.PersonId=Address.PersonId;


# 176. Second Highest Salary
# Write your MySQL query statement below
# Select Max(Salary)  AS SecondHighestSalary
# From Employee Where Salary <
# (Select Max(Salary) From Employee)
# CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
# BEGIN
#   RETURN (
#       # Write your MySQL query statement below.
#       Select Max(Salary)  From Employee E1
#       Where (N-1) =
#       (Select Count(Distinct(E2.Salary)) From Employee E2
#       Where E2.Salary>E1.Salary)
#   );
# END


# 177. Nth Highest Salary
# CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
# BEGIN
#   RETURN (
#       # Write your MySQL query statement below.
#       Select Max(Salary)  From Employee E1
#       Where (N-1) =
#       (Select Count(Distinct(E2.Salary)) From Employee E2
#       Where E2.Salary>E1.Salary)
#   );
# END


# 178. Rank Scores
# Write your MySQL query statement below
# Select Score,
# (Select Count(Distinct Score) From Scores Where Score >= s.Score) Rank
# From Scores s Order By Score Desc


# 179. Largest Number
class Solution(object):
    def largestNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: str
        """
        def compare(a, b):
            if (a+b)>(b+a):
                return -1
            else:
                return 1
        data = sorted([str(num) for num in nums], cmp=compare)
        result = "".join(data).lstrip("0")
        if result == "":
            result = "0"
        return result
        # res = solu.largestNumber([3, 30, 34, 5, 9])
# 180. Consecutive Numbers
# Write your MySQL query statement below
# Select Distinct l1.Num  As ConsecutiveNums From Logs l1
# Join Logs l2 On l1.Id = l2.Id - 1
# Join Logs l3 On l1.Id = l3.Id - 2
# Where l1.Num = l2.Num And l2.Num = l3.Num
if __name__ == '__main__':
    solu = Solution()
    test = [[-2, -3, 3], [-5, -10, 1], [10, 30, -5]]
    res = solu.largestNumber([3, 30, 34, 5, 9])
    print res
