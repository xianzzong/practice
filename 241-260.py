# -*- coding: utf-8 -*
# 241. Different Ways to Add Parentheses


class Solution(object):
    def diffWaysToCompute(self, input):
        """
        :type input: str
        :rtype: List[int]
        """
        res = []
        length = len(input)
        for i in range(length):
            if input[i] in "+-*":
                left = self.diffWaysToCompute(input[:i])
                right = self.diffWaysToCompute(input[i + 1:])
                for l in left:
                    for r in right:
                        if input[i] == "+":
                            res.append(l + r)
                        elif input[i] == "-":
                            res.append(l - r)
                        elif input[i] == "*":
                            res.append(l * r)
        if len(res) == 0:
            res.append(int(input))
        return res
# res = solu.diffWaysToCompute("2*3-4*5")
# 242. Valid Anagram


class Solution(object):
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        if len(s) != len(t):
            return False
        count = {}
        for i in range(len(s)):
            if s[i] not in count:
                count[s[i]] = 1
            else:
                count[s[i]] += 1
            if t[i] not in count:
                count[t[i]] = -1
            else:
                count[t[i]] -= 1
        for key, value in count.iteritems():
            if value != 0:
                return False
        return True


# 257. Binary Tree Paths
class Solution(object):
    def binaryTreePaths(self, root):
        """
        :type root: TreeNode
        :rtype: List[str]
        """
        res = []
        if root is None:
            return []
        if root.left is None and root.right is None:
            return [str(root.val)]
        if root.left:
            left = self.binaryTreePaths(root.left)
            for item in left:
                res.append("{}->{}".format(root.val, item))
        if root.right:
            right = self.binaryTreePaths(root.right)
            for item in right:
                res.append("{}->{}".format(root.val, item))
        return res


# 258. Add Digits
class Solution(object):
    def addDigits(self, num):
        """
        :type num: int
        :rtype: int
        """
        res = 0
        if num < 10:
            return num
        while True:
            while num > 0:
                res += num % 10
                num = int(num / 10)
            if res < 10:
                break
            else:
                num = res
                res = 0
        return res


# res = solu.addDigits(38)

# 260. Single Number III
class Solution(object):
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        # 首先计算nums数组中所有数字的异或，记为xor
        # 令lowbit = xor & -xor，lowbit的含义为xor从低位向高位，第一个非0位所对应的数字
        # 例如假设xor = 6（二进制：0110），则- xor为（二进制：1010，- 6的补码，two's complement）
        # 则lowbit = 2（二进制：0010）
        # 根据异或运算的性质，“同0异1”
        # 记只出现一次的两个数字分别为a与b
        # 可知a & lowbit与b & lowbit的结果一定不同
        # 通过这种方式，即可将a与b拆分开来
        xor = 0
        for num in nums:
            xor = xor ^ num
        lowbit = xor & -xor
        a = b = 0
        for num in nums:
            if num & lowbit:
                a = a ^ num
            else:
                b = b ^ num
        return [a, b]


if __name__ == "__main__":
    solu = Solution()
    res = solu.singleNumber([1, 2, 1, 3, 2, 5])
    print(res)
