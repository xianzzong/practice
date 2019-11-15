# 181. Employees Earning More Than Their Managers
# Write your MySQL query statement below
# Select e1.Name As Employee From Employee e1, Employee e2
# Where e1.ManagerId = e2.Id and e1.Salary > e2.Salary

# 182. Duplicate Emails
# Write your MySQL query statement below
# Select Email from Person Group by Email
# Having Count(*) > 1

# 183. Customers Who Never Order
# Select Name As Customers From Customers
# Where Id not in (Select CustomerId From Orders)


# 184. Department Highest Salary
# Select d.Name As Department, e.Name As Employee, e.Salary From Employee e, Department d
# Where e.DepartmentId = d.Id And e.Salary = (Select Max(Salary) From Employee e2
#                                            Where e2.DepartmentId = d.Id)


# 185. Department Top Three Salaries
# Select d.Name As Department, e.Name as Employee, e.Salary From Employee e
# Join Department d on e.DepartmentId = d.Id
# Where (Select Count(Distinct Salary) From Employee Where Salary > e.Salary
# And DepartmentId = d.Id) < 3 Order By d.Name, e.Salary Desc;

# 187. Repeated DNA Sequences
class Solution(object):
    def findRepeatedDnaSequences(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        res = []
        result = {}
        length = len(s)
        sum = 0
        map = {"A": 0, "C": 1, "G": 2, "T": 3}
        if length < 10:
            return res
        for i in range(length):
            sum = (sum * 4 + map[s[i]]) & 0xFFFFF
            if i < 9:
                continue
            result[sum] = result.get(sum, 0) + 1
            if result[sum] == 2:
                res.append(s[i - 9:i + 1])
        return res
        # res = solu.findRepeatedDnaSequences("AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT")

# 188. Best Time to Buy and Sell Stock IV


class Solution(object):
    def maxProfit(self, k, prices):
        """
        :type k: int
        :type prices: List[int]
        :rtype: int
        """
        len_p = len(prices)
        result = 0
        if 2 * k > len_p:
            for i in range(len_p - 1):
                if prices[i + 1] > prices[i]:
                    result += prices[i + 1] - prices[i]
        else:
            tmp = [None] * (2 * k + 1)
            tmp[0] = 0
            for i in range(len_p):
                for j in range(min(2 * k, i + 1), 0, -1):
                    tmp[j] = max(tmp[j], tmp[j - 1] +
                                 prices[i] * [1, -1][j % 2])
            result = max(tmp)
        return result

# 189. Rotate Array


class Solution(object):
    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        def reverse(nums, start, end):
            while start < end:
                nums[start], nums[end] = nums[end], nums[start]
                start += 1
                end -= 1
        len_n = len(nums)
        k = k % len_n
        if k == 0:
            return nums
        reverse(nums, 0, len_n - 1)
        reverse(nums, 0, k - 1)
        reverse(nums, k, len_n - 1)
        return nums
# res = solu.rotate([1, 2, 3, 4, 5, 6, 7], 0)

# 190. Reverse Bits


class Solution:
    # @param n, an integer
    # @return an integer
    def reverseBits(self, n):
        tmp = bin(n)
        tmp = tmp[2:]
        tmp = '0' * (32 - len(tmp)) + tmp
        tmp = tmp[::-1]
        return int(tmp, 2)

# 191. Number of 1 Bits


class Solution(object):
    def hammingWeight(self, n):
        """
        :type n: int
        :rtype: int
        """
        res = 0
        while n:
            res += n & 1
            n = n >> 1
        return res


# 192. Word Frequency
# cat words.txt |tr -s ' ' '\n'|sort|uniq -c|sort -rn |awk '{print $2,$1}'

# 193. Valid Phone Numbers
# # Read from the file file.txt and output all valid phone numbers to stdout.
# awk '/^([0-9]{3}-|\([0-9]{3}\) )[0-9]{3}-[0-9]{4}$/' file.txt

# 194. Transpose File
# Read from the file file.txt and print its transposed content to stdout.
# awk '{
#     for (i = 1; i <= NF; ++i) {
#         if (NR == 1) s[i] = $i;
#         else s[i] = s[i] " " $i;
#     }
# } END {
#     for (i = 1; s[i] != ""; ++i) {
#         print s[i];
#     }
# }' file.txt

# 195. Tenth Line
# # Read from the file file.txt and output the tenth line to stdout.
# awk '{if(NR == 10) print $0}' file.txt

# 196. Delete Duplicate Emails
# DELETE p1 FROM Person p1,
#     Person p2
# WHERE
#     p1.Email = p2.Email AND p1.Id > p2.Id

# 197. Rising Temperature

# # Write your MySQL query statement below
# SELECT w1.Id FROM Weather w1, Weather w2
# WHERE w1.Temperature > w2.Temperature AND DATEDIFF(w1.RecordDate, w2.RecordDate) = 1;

# 198. House Robber
class Solution(object):
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        length = len(nums)
        if length == 0:
            return 0
        if length <= 2:
            return max(nums)
        result = [0 for i in range(length)]
        result[0] = nums[0]
        result[1] = nums[1]
        for i in range(2, length, 1):
            result[i] = max(result[i - 1], max(result[:i - 1]) + nums[i])
        print result
        return result[length - 1]
        # res = solu.rob([2, 1, 1, 2])

# 199. Binary Tree Right Side View


class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution(object):
    def rightSideView(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        result = []
        if root is None:
            return result
        tmp_node = [root]
        while tmp_node:
            length = len(tmp_node)
            for i in range(length):
                tmp = tmp_node.pop(0)
                if i == 0:
                    result.append(tmp.val)
                if tmp.right:
                    tmp_node.append(tmp.right)
                if tmp.left:
                    tmp_node.append(tmp.left)
        return result

# 200. Number of Islands


class Solution(object):
    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        def change(x, y):
            if x < 0 or x >= row or y < 0 or y >= col:
                return
            else:
                if grid[x][y] == "1":
                    grid[x][y] = "0"
                    change(x - 1, y)
                    change(x + 1, y)
                    change(x, y - 1)
                    change(x, y + 1)
            return
        num_island = 0
        row = len(grid)
        if row == 0:
            return 0
        col = len(grid[0])
        if col == 0:
            return 0
        for i in range(row):
            for j in range(col):
                if grid[i][j] == "1":
                    num_island += 1
                    change(i, j)
        return num_island


if __name__ == "__main__":
    solu = Solution()
    test = [["1", "1", "0", "0", "0"],
            ["1", "1", "0", "0", "0"],
            ["0", "0", "1", "0", "0"],
            ["0", "0", "1", "1", "1"]]
    res = solu.numIslands(test)
    print res
