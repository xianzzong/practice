class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
# 61. Rotate List


class Solution(object):
    def rotateRight(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        if k == 0 or head is None:
            return head
        tmp = ListNode(0)
        tmp.next = head
        p = tmp
        num = 0
        while p.next:
            p = p.next
            num += 1
        print num
        step = num - k % num
        p.next = tmp.next
        for i in range(step):
            p = p.next
        head = p.next
        p.next = None
        return head

# test
#a = ListNode(1)
#b = ListNode(2)
#c = ListNode(3)
#d = ListNode(4)
#e = ListNode(5)
#a.next = b
#b.next = c
#c.next = d
#d.next = e
#e.next = None

# 62. Unique Paths


class Solution(object):
    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        def A_n(n):
            res = 1
            while n > 1:
                res = res * n
                n -= 1
            return res
        if m == 1 or n == 1:
            return 1
        return A_n(m + n - 2) / (A_n(m - 1) * A_n(n - 1))
#test = [7, 3]

# 63. Unique Paths II


class Solution(object):
    def uniquePathsWithObstacles(self, obstacleGrid):
        """
        :type obstacleGrid: List[List[int]]
        :rtype: int
        """
        row = len(obstacleGrid)
        col = len(obstacleGrid[0])
        result = [[0 for i in range(col)] for j in range(row)]
        if obstacleGrid[0][0] == 1:
            return result[0][0]
        else:
            result[0][0] = 1
        for i in range(1, row, 1):
            if obstacleGrid[i][0] == 1:
                result[i][0] = 0
                break
            else:
                result[i][0] = 1
        for i in range(1, col, 1):
            if obstacleGrid[0][i] == 1:
                result[0][i] = 0
                break
            else:
                result[0][i] = 1
        for i in range(1, row, 1):
            for j in range(1, col, 1):
                if obstacleGrid[i][j] == 1:
                    result[i][j] = 0
                else:
                    result[i][j] = result[i - 1][j] + result[i][j - 1]
        return result
#test = [[0, 1, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
# 64. Minimum Path Sum


class Solution(object):
    def minPathSum(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        row = len(grid)
        col = len(grid[0])
        result = [[0 for i in range(col)] for j in range(row)]
        result[0][0] = grid[0][0]
        for i in range(1, row, 1):
            result[i][0] = result[i - 1][0] + grid[i][0]
        for i in range(1, col, 1):
            result[0][i] = result[0][i - 1] + grid[0][i]
        for i in range(1, row, 1):
            for j in range(1, col, 1):
                result[i][j] = min(
                    result[i - 1][j], result[i][j - 1]) + grid[i][j]
        return result[row - 1][col - 1]
# test = [
#[1, 3, 1],
#[1, 5, 1],
#[4, 2, 1]
# ]
# 65. Valid Number


class Solution(object):
    def isNumber(self, s):
        """
        :type s: str
        :rtype: bool
        """
        s = s.strip()
        begin = 0
        if begin < len(s) and (s[begin] == '+' or s[begin] == '-'):
            begin += 1
        flag_number, flag_exp, flag_dot = False, False, False
        while begin < len(s):
            if s[begin] >= '0' and s[begin] <= '9':
                flag_number = True
            elif s[begin] == '.':
                if flag_exp or flag_dot:
                    return False
                else:
                    flag_dot = True
            elif s[begin] == 'e' or s[begin] == 'E':
                if flag_exp or not flag_number:
                    return False
                else:
                    flag_exp = True
                    flag_number = False
            elif s[begin] == '+' or s[begin] == '-':
                if s[begin - 1] != 'e' and s[begin - 1] != 'E':
                    return False
            else:
                return False
            begin += 1
        return flag_number

# 66. Plus One


class Solution(object):
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        flag = 0
        length = len(digits) - 1
        digits[length] += 1
        for i in range(length, -1, -1):
            digits[i] += flag
            flag = 0
            if digits[i] > 9:
                flag = 1
                digits[i] = digits[i] % 10
            if flag == 0:
                break
        if flag:
            digits.insert(0, 1)
        return digits
        #test = [1, 2, 3]
# 67. Add Binary


class Solution(object):
    def addBinary(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """
        max_length = max(len(a), len(b))
        a = a.zfill(max_length)
        b = b.zfill(max_length)
        flag = 0
        res = ''
        for i in range(max_length - 1, -1, -1):
            tmp = int(a[i]) + int(b[i]) + flag
            if tmp > 1:
                res += str(tmp % 2)
                flag = 1
            else:
                res += str(tmp)
                flag = 0
        if flag:
            res += '1'
        return res[::-1]
        # test = ['1010', '1011']

# 68. Text Justification


class Solution(object):
    def fullJustify(self, words, maxWidth):
        """
        :type words: List[str]
        :type maxWidth: int
        :rtype: List[str]
        """
        item_index = 0
        result = []
        tmp = []
        length = 0
        while item_index < len(words):
            length += len(words[item_index])
            if length <= maxWidth:
                length += 1
                tmp.append(item_index)
                item_index += 1
            else:
                length = 0
                result.append(tmp)
                tmp = []
        result.append(tmp)
        len_res = len(result)
        for j in range(len_res - 1):
            blank = maxWidth
            for i in range(len(result[j])):
                blank -= len(words[result[j][i]])
            if len(result[j]) > 1:
                space_num = blank / (len(result[j]) - 1)
                space_num_1 = blank % (len(result[j]) - 1)
                tmp = words[result[j][0]]
                for i in range(1, len(result[j]), 1):
                    if space_num_1 > 0:
                        tmp += " " * (space_num + 1) + words[result[j][i]]
                        space_num_1 -= 1
                    else:
                        tmp += " " * (space_num) + words[result[j][i]]
            else:
                tmp = words[result[j][0]] + blank * " "
            result[j] = tmp
        tmp = words[result[len_res - 1][0]]
        for i in range(1, len(result[len_res - 1]), 1):
            tmp += " " + words[result[len_res - 1][i]]
        space_num = maxWidth - len(tmp)
        tmp += space_num * " "
        result[len_res - 1] = tmp
        return result
        #test = ["What", "must", "be", "acknowledgment", "shall", "be"]

# 69. Sqrt(x)


class Solution(object):
    def mySqrt(self, x):
        """
        :type x: int
        :rtype: int
        """
        if x <= 1:
            return x
        left = 1
        right = x
        while (right - left) > 1:
            mid = (right + left) / 2
            tmp = mid ** 2
            if tmp == x:
                return mid
            elif tmp < x:
                left = mid
            else:
                right = mid
        return left

# 70. Climbing Stairs


class Solution(object):
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        res = [1, 2]
        for i in range(2, n):
            res.append(res[i - 1] + res[i - 2])
        return res[n - 1]

# 71. Simplify Path


class Solution(object):
    def simplifyPath(self, path):
        """
        :type path: str
        :rtype: str
        """
        path = path.strip("/")
        res = []
        path = path.split("/")
        for item in path:
            if item == '..':
                if len(res) > 0:
                    res.pop()
                else:
                    continue
            elif item == '' or item == '.' or item == '/':
                continue
            else:
                res.append(item)
        res = '/'.join(res)
        return '/' + res
# 72. Edit Distance


class Solution(object):
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        len_1 = len(word1) + 1
        len_2 = len(word2) + 1
        res = [[0 for i in range(len_2)]for j in range(len_1)]
        for i in range(len_1):
            res[i][0] = i
        for i in range(len_2):
            res[0][i] = i
        for i in range(1, len_1, 1):
            for j in range(1, len_2, 1):
                if word1[i - 1] == word2[j - 1]:
                    res[i][j] = res[i - 1][j - 1]
                else:
                    res[i][j] = min(res[i - 1][j], res[i]
                                    [j - 1], res[i - 1][j - 1]) + 1
        return res[len_1 - 1][len_2 - 1]

# 73. Set Matrix Zeroes


class Solution(object):
    def setZeroes(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """
        row = len(matrix)
        col = len(matrix[0])
        is_col = False
        for i in range(row):
            if matrix[i][0] == 0:
                is_col = True
            for j in range(1, col, 1):
                if matrix[i][j] == 0:
                    matrix[i][0] = 0
                    matrix[0][j] = 0
        for i in range(1, row, 1):
            for j in range(1, col, 1):
                if matrix[i][0] == 0 or matrix[0][j] == 0:
                    matrix[i][j] = 0
        if matrix[0][0] == 0:
            for j in range(col):
                matrix[0][j] = 0
        if is_col:
            for i in range(row):
                matrix[i][0] = 0
        return
        #     test = [
        #   [0,1,2,0],
        #   [3,4,5,2],
        #   [1,3,1,5]
        # ]

# 74. Search a 2D Matrix


class Solution(object):
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        row = len(matrix)
        if row == 0:
            return False
        col = len(matrix[0])
        if col == 0:
            return False
        first = 0
        last = row - 1
        while first < last:
            mid = (first + last + 1) / 2
            if matrix[mid][0] == target:
                return True
            elif matrix[mid][0] < target:
                first = mid
            else:
                last = mid - 1
        target_row = first
        first = 0
        last = col - 1
        while first <= last:
            mid = (first + last) / 2
            if matrix[target_row][mid] == target:
                return True
            elif matrix[target_row][mid] < target:
                first = mid + 1
            else:
                last = mid - 1
        return False
        # test = [
        #     [1,   3,  5,  7],
        #     [10, 11, 16, 20],
        #     [23, 30, 34, 50]
        # ]
# 75. Sort Colors


class Solution(object):
    def sortColors(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        left = 0
        right = len(nums) - 1
        i = 0
        while i <= right:
            if nums[i] == 0:
                nums[i], nums[left] = nums[left], nums[i]
                left += 1
                i += 1
            elif nums[i] == 2:
                nums[i], nums[right] = nums[right], nums[i]
                right -= 1
            else:
                i += 1
        return test
        #test = [2,0,1]

# 76. Minimum Window Substring


class Solution(object):
    def minWindow(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        dict_t = {}
        len_t = len(t)
        for item in t:
            if item in dict_t:
                dict_t[item] += 1
            else:
                dict_t[item] = 1
        start = 0
        min_length = len(s) + 1
        res = ""
        for end in range(len(s)):
            if s[end] in dict_t:
                dict_t[s[end]] -= 1
                if dict_t[s[end]] >= 0:
                    len_t -= 1
            if len_t == 0:
                while True:
                    if s[start] not in dict_t:
                        start += 1
                    else:
                        if dict_t[s[start]] < 0:
                            dict_t[s[start]] += 1
                            start += 1
                        else:
                            break
                sub_str = s[start:end + 1]
                if len(sub_str) <= min_length:
                    min_length = len(sub_str)
                    res = sub_str
        return res
        #test = "ADOBECODEBANC"

# 77. Combinations


class Solution(object):
    def combine(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[List[int]]
        """
        def solve_com(start, n, k):
            res = []
            if k == n - start + 1:
                tmp = []
                for item in range(k):
                    tmp.append(start)
                    start += 1
                res.append(tmp)
                return res
            if k == 1:
                while start <= n:
                    res.append([start])
                    start += 1
                return res
            tmp1 = solve_com(start + 1, n, k - 1)
            tmp2 = solve_com(start + 1, n, k)
            for item in tmp1:
                tmp = item + [start]
                res.append(tmp)
            for item in tmp2:
                res.append(item)
            return res
        if k > n or k <= 0:
            return [[]]
        res = solve_com(1, n, k)
        return res
        #test  [4, 2]

# 78. Subsets


class Solution(object):
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        def solve(nums):
            res = []
            length = len(nums)
            if length == 0:
                return [[]]
            tmp = solve(nums[:length - 1])
            for item in tmp:
                res.append(item)
                res.append(item + [nums[length - 1]])
            return res
        res = solve(nums)
        return res
# 79. Word Search


class Solution(object):
    def exist(self, board, word):
        """
        :type board: List[List[str]]
        :type word: str
        :rtype: bool
        """
        row = len(board)
        col = len(board[0])

        def solve(x, y, items):
            if len(items) == 0:
                return True
            # go up
            if x > 0 and board[x - 1][y] == items[0]:
                tmp = board[x][y]
                board[x][y] = "*"
                if solve(x - 1, y, items[1:]):
                    return True
                else:
                    board[x][y] = tmp
            # go down
            if x < row - 1 and board[x + 1][y] == items[0]:
                tmp = board[x][y]
                board[x][y] = "*"
                if solve(x + 1, y, items[1:]):
                    return True
                else:
                    board[x][y] = tmp
            # go right
            if y > 0 and board[x][y - 1] == items[0]:
                tmp = board[x][y]
                board[x][y] = "*"
                if solve(x, y - 1, items[1:]):
                    return True
                else:
                    board[x][y] = tmp
            # go left
            if y < col - 1 and board[x][y + 1] == items[0]:
                tmp = board[x][y]
                board[x][y] = "*"
                if solve(x, y + 1, items[1:]):
                    return True
                else:
                    board[x][y] = tmp
            return False
        for i in range(row):
            for j in range(col):
                if board[i][j] == word[0]:
                    if solve(i, j, word[1:]):
                        return True
        return False
        # test = [
        #     ['A', 'B', 'C', 'E'],
        #     ['S', 'F', 'C', 'S'],
        #     ['A', 'D', 'E', 'E']
        # ]
# 80. Remove Duplicates from Sorted Array II
class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        i = 1
        flag = 1
        while i < len(nums):
            if nums[i] == nums[i-1]:
                flag += 1
                if flag > 2:
                    nums.remove(nums[i])
                    flag -= 1
                else:
                    i += 1
            else:
                flag = 1
                i += 1
        return len(nums)




if __name__ == '__main__':
    solu = Solution()
    test = [1,1,1,2,2,3]
    res = solu.removeDuplicates(test)
    print res
