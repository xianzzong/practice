# 41. First Missing Positive
class Solution(object):
    def firstMissingPositive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        len_nums = len(nums)
        i = 0
        while i < len_nums:
            if 1 <= nums[i] <= len_nums and nums[i] != nums[nums[i] - 1]:
                tmp = nums[i] - 1
                nums[i], nums[tmp] = nums[tmp], nums[i]
            else:
                i += 1
        for i in range(len_nums):
            if nums[i] != i + 1:
                return i + 1
        return len_nums + 1
        #test=[3, 4, -1, 1]

# 42. Trapping Rain Water


class Solution(object):
    def trap(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        left = 0
        right = len(height) - 1
        res = 0
        left_max = 0
        right_max = 0
        while left < right:
            if height[left] < height[right]:
                if height[left] >= left_max:
                    left_max = height[left]
                else:
                    res += left_max - height[left]
                left += 1
            else:
                if height[right] >= right_max:
                    right_max = height[right]
                else:
                    res += right_max - height[right]
                right -= 1
        return res
        #test = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]

# 43. Multiply Strings


class Solution(object):
    def multiply(self, num1, num2):
        """
        :type num1: str
        :type num2: str
        :rtype: str
        """
        num1 = num1[::-1]
        num2 = num2[::-1]
        res = 0
        for i in range(len(num1)):
            tmp_sum = 0
            for j in range(len(num2)):
                tmp_sum += 10 ** j * int(num1[i]) * int(num2[j])
            res += 10 ** i * tmp_sum
        return str(res)
        #test = ['456', '123']

# 44. Wildcard Matching


class Solution(object):
    def isMatch(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: bool
        """
        len_s = len(s)
        len_p = len(p)
        s_index = 0
        p_index = 0
        flag = -1
        s_tmp_index = 0
        while s_index < len(s):
            if p_index < len(p) and (s[s_index] == p[p_index] or p[p_index] == "?"):
                s_index += 1
                p_index += 1
                continue
            if p_index < len(p) and (p[p_index] == "*"):
                flag = p_index
                p_index += 1
                s_tmp_index = s_index
                continue
            if flag != -1:
                p_index = flag + 1
                s_tmp_index += 1
                s_index = s_tmp_index
                continue
            return False
        while p_index < len(p):
            if p[p_index] != "*":
                return False
            p_index += 1
        if p_index == len(p):
            return True
        return Falses

# 45. Jump Game II


class Solution(object):
    def jump(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        len_n = len(nums)
        current = 0
        last = 0
        res = 0
        for i in range(len_n):
            if i > last:
                last = current
                res += 1
                if last > len_n:
                    break
            current = max(current, i + nums[i])
        return res
        #test = [2,3,1,1,4]

# 46. Permutations


class Solution(object):
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        if len(nums) == 0:
            return []
        if len(nums) == 1:
            return [nums]
        res = []
        for i in range(len(nums)):
            tmp = self.permute(nums[:i] + nums[i + 1:])
            for j in tmp:
                res.append(j + [nums[i]])
        return res

        #test = [1, 2, 3]

# 47. Permutations II


class Solution(object):
    def permuteUnique(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        nums.sort()
        if len(nums) == 0:
            return []
        if len(nums) == 1:
            return [nums]
        res = []
        i = 0
        while i < len(nums):
            if i > 0 and nums[i] == nums[i - 1]:
                i += 1
                continue
            tmp = self.permuteUnique(nums[:i] + nums[i + 1:])
            for j in tmp:
                res.append(j + [nums[i]])
            i += 1
        return res
        #test = [1, 2, 1]

# 48. Rotate Image


class Solution(object):
    def rotate(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """
        number = len(matrix)
        for i in range(number):
            for j in range(i + 1, number, 1):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        for i in range(number):
            matrix[i].reverse()
        return
    # test = [
    #     [1, 2, 3],
    #     [4, 5, 6],
    #     [7, 8, 9]
    # ]

# 49. Group Anagrams


class Solution(object):
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        res = {}
        for item in strs:
            tmp = ''.join(sorted(item))
            if tmp not in res:
                res[tmp] = [item]
            else:
                res[tmp].append(item)
        return res.values()
        #test = ["eat", "tea", "tan", "ate", "nat", "bat"]

# 50. Pow(x, n)


class Solution(object):
    def myPow(self, x, n):
        """
        :type x: float
        :type n: int
        :rtype: float
        """
        if n == 0:
            return 1
        if n == 1:
            return x
        if n < 0:
            return 1 / self.myPow(x, -n)
        if n > 0:
            if n % 2 == 0:
                return self.myPow(x * x, n / 2)
            else:
                return x * self.myPow(x * x, n / 2)

# 51. N-Queens


class Solution(object):
    def solveNQueens(self, n):
        """
        :type n: int
        :rtype: List[List[str]]
        """
        def isValid(row, col):
            for i in range(row):
                if judger[i] == col or abs(row - i) == abs(col - judger[i]):
                    return False
            return True

        def fillQueen(row, row_value):
            if row == n:
                res.append(row_value)
                return
            for col in range(n):
                if isValid(row, col):
                    judger[row] = col
                    fillQueen(row + 1, row_value +
                              ["." * col + "Q" + "." * (n - 1 - col)])
        res = []
        judger = [-1 for i in range(n)]
        fillQueen(0, [])
        return res

# 52. N-Queens II


class Solution(object):
    def totalNQueens(self, n):
        """
        :type n: int
        :rtype: int
        """
        res = []
        judger = [-1 for i in range(n)]

        def isValid(row, col):
            for i in range(row):
                if judger[i] == col or abs(i - row) == abs(col - judger[i]):
                    return False
            return True

        def fillQueen(row):
            if row == n:
                res.append(1)
                return
            for col in range(n):
                if isValid(row, col):
                    judger[row] = col
                    fillQueen(row + 1)
        fillQueen(0)
        return len(res)


# 53. Maximum Subarray
class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        max_sum = nums[0]
        cur_sum = 0
        for i in range(len(nums)):
            if cur_sum < 0:
                cur_sum = 0
            cur_sum += nums[i]
            max_sum = max(cur_sum, max_sum)
        return max_sum
        #test = [-2]

# 54. Spiral Matrix


class Solution(object):
    def spiralOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        res = []
        if len(matrix) == 0 or len(matrix[0]) == 0:
            return []
        go_right = 0
        go_down = 0
        go_left = len(matrix[0]) - 1
        go_up = len(matrix) - 1
        direct = 0
        while True:
            if direct == 0:
                for i in range(go_right, go_left + 1, 1):
                    res.append(matrix[go_down][i])
                go_down += 1
            if direct == 1:
                for i in range(go_down, go_up + 1, 1):
                    res.append(matrix[i][go_left])
                go_left -= 1
            if direct == 2:
                for i in range(go_left, go_right - 1, -1):
                    res.append(matrix[go_up][i])
                go_up -= 1
            if direct == 3:
                for i in range(go_up, go_down - 1, -1):
                    res.append(matrix[i][go_right])
                go_right += 1
            direct = (direct + 1) % 4
            if go_right > go_left or go_down > go_up:
                return res
        #     test = [
        #  [ 1, 2, 3 ],
        #  [ 4, 5, 6 ],
        #  [ 7, 8, 9 ]
        # ]

# 55. Jump Game


class Solution(object):
    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """

        if len(nums) <= 1:
            return True
        current = nums[0]
        for i in range(1, len(nums), 1):
            if current < i:
                return False
            if nums[i] == 0 and current <= i and i != len(nums) - 1:
                return False
            current = max(current, i + nums[i])
            if current >= len(nums) - 1:
                return True
            #test = [2, 0,0]
# 56. Merge Intervals


class Solution(object):
    def merge(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: List[List[int]]
        """
        res = []
        if len(intervals) <= 1:
            return intervals
        intervals.sort(key=lambda x: x[0])
        tmp = intervals[0]
        for i in range(1, len(intervals), 1):
            if tmp[1] >= intervals[i][0]:
                tmp[1] = max(tmp[1], intervals[i][1])
            else:
                res.append(tmp)
                tmp = intervals[i]
        res.append(tmp)
        return res
        #test = [[1, 3], [2, 6], [8, 10], [15, 18]]

# 57. Insert Interval


class Solution(object):
    def insert(self, intervals, newInterval):
        """
        :type intervals: List[List[int]]
        :type newInterval: List[int]
        :rtype: List[List[int]]
        """
        if len(intervals) <= 0:
            return [newInterval]
        res = []
        intervals.sort(key=lambda x: x[0])
        tmp = intervals[0]
        flag = True
        if tmp[0] >= newInterval[0]:
            tmp = newInterval
            flag = False
        i = 0
        while i < len(intervals):
            if flag and intervals[i][0] >= newInterval[0]:
                flag = False
                if tmp[1] >= newInterval[0]:
                    tmp[1] = max(tmp[1], newInterval[1])
                else:
                    res.append(tmp)
                    tmp = newInterval
            elif tmp[1] >= intervals[i][0]:
                tmp[1] = max(tmp[1], intervals[i][1])
                i += 1
            else:
                res.append(tmp)
                tmp = intervals[i]
                i += 1
        if flag:
            if tmp[1] >= newInterval[0]:
                tmp[1] = max(tmp[1], newInterval[1])
                res.append(tmp)
            else:
                res.append(tmp)
                res.append(newInterval)
        else:
            res.append(tmp)
        return res
        # test = [[1,3],[6,9]] test2= [[1,3],[6,9]]

# 58. Length of Last Word


class Solution(object):
    def lengthOfLastWord(self, s):
        """
        :type s: str
        :rtype: int
        """
        tmp = s.strip().split()
        if len(tmp) == 0:
            return 0
        else:
            return len(tmp[-1])

# 59. Spiral Matrix II


class Solution(object):
    def generateMatrix(self, n):
        """
        :type n: int
        :rtype: List[List[int]]
        """
        if n <= 0:
            return [[]]
        res = [[0 for i in range(n)] for i in range(n)]
        direct = 0
        go_left = 0
        go_right = n - 1
        go_down = 0
        go_up = n - 1
        i = 1
        while i <= n * n:
            if direct == 0:
                for j in range(go_left, go_right + 1, 1):
                    res[go_down][j] = i
                    i += 1
                go_down += 1
                direct += 1
            if direct == 1:
                for j in range(go_down, go_up + 1, 1):
                    res[j][go_right] = i
                    i += 1
                go_right -= 1
                direct += 1
            if direct == 2:
                for j in range(go_right, go_left - 1, -1):
                    res[go_up][j] = i
                    i += 1
                go_up -= 1
                direct += 1
            if direct == 3:
                for j in range(go_up, go_down - 1, -1):
                    res[j][go_left] = i
                    i += 1
                go_left += 1
                direct += 1
            direct = direct % 4
        return res
# 60. Permutation Sequence


class Solution(object):
    def getPermutation(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: str
        """
        nums = [i + 1 for i in range(n)]
        f_n = 1
        res = ""
        for i in range(1, n, 1):
            f_n = f_n * i
        k = k - 1
        for i in range(n - 1, -1, -1):
            curr = nums[k / f_n]
            res += str(curr)
            nums.remove(curr)
            if i != 0:
                k = k % f_n
                f_n = f_n / i
        return res


if __name__ == '__main__':
    solu = Solution()
    test = [3, 3]

    ans = solu.getPermutation(test[0], test[1])
    print ans
