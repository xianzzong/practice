# -*- coding: utf-8 -*
# 263. Ugly Number


class Solution(object):
    def isUgly(self, num):
        """
        :type num: int
        :rtype: bool
        """
        while num >= 2:
            if num % 2 == 0:
                num = num / 2
            elif num % 3 == 0:
                num = num / 3
            elif num % 5 == 0:
                num = num / 5
            else:
                return False
        return num == 1


# 262. Trips and Users
# SELECT Request_at Day, ROUND(COUNT(IF(Status != 'completed', TRUE, NULL)) / COUNT(*), 2) 'Cancellation Rate'
# FROM Trips WHERE(Request_at BETWEEN '2013-10-01' AND '2013-10-03') AND Client_Id IN
# (SELECT Users_Id FROM Users WHERE Banned='No') GROUP BY Request_at
# 264. Ugly Number II


class Solution(object):
    def nthUglyNumber(self, n):
        """
        :type n: int
        :rtype: int
        """
        n2, n3, n5 = 0, 0, 0
        res = [1] * n
        for i in range(1, n, 1):
            res[i] = min(res[n2] * 2, res[n3] * 3, res[n5] * 5)
            if res[i] == res[n2] * 2:
                n2 += 1
            if res[i] == res[n3] * 3:
                n3 += 1
            if res[i] == res[n5] * 5:
                n5 += 1
        return res[n-1]


# 268. Missing Number
class Solution(object):
    def missingNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        length = len(nums)
        res = 0
        for i in range(length + 1):
            if i == length:
                res = res ^ i
            else:
                res = res ^ i
                res = res ^ nums[i]
        return res


# res = solu.missingNumber([3, 0, 1])

# 273. Integer to English Words
class Solution(object):
    def numberToWords(self, num):
        """
        :type num: int
        :rtype: str
        """
        lv1 = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten', 'Eleven',
               'Twelve', 'Thirteen', 'Fourteen', 'Fifteen', 'Sixteen', 'Seventeen', 'Eighteen', 'Nineteen']
        lv2 = ['Twenty', 'Thirty', 'Forty', 'Fifty', 'Sixty', 'Seventy', 'Eighty', 'Ninety']
        lv3 = "Hundred"
        lv4 = ['Thousand', 'Million', 'Billion']
        words = []
        digits = 0
        while num:
            token, num = num % 1000, int(num / 1000)
            word = ""
            if token > 99:
                word += lv1[int(token / 100)] + " " + lv3 + " "
                token = token % 100
            if token > 19:
                word += lv2[int(token / 10) - 2] + " "
                token = token % 10
            if token > 0:
                word += lv1[token] + " "
            word = word.strip()
            if word:
                word += ' ' + lv4[digits - 1] if digits else ''
                words.append(word)
            digits += 1
        return " ".join(words[::-1]) or "Zero"


# res = solu.numberToWords(121)

# 274. H-Index
class Solution(object):
    def hIndex(self, citations):
        """
        :type citations: List[int]
        :rtype: int
        """
        if len(citations) <= 0:
            return 0
        citations = sorted(citations, reverse=True)
        res = 0
        for i, c in enumerate(citations):
            res = max(min(i+1, c), res)
        return res


# res = solu.hIndex([3, 0, 6, 1, 5])

# 275. H-Index II


class Solution(object):
    def hIndex(self, citations):
        """
        :type citations: List[int]
        :rtype: int
        """
        left, right, lenth = 0, len(citations), len(citations)
        while left < right:
            mid = (left + right) >> 1
            if lenth - mid <= citations[mid]:
                right = mid
            else:
                left = mid + 1
        return lenth - left


# res = solu.hIndex([0, 1, 3, 5, 6])

# 278. First Bad Version
class Solution(object):
    def firstBadVersion(self, n):
        """
        :type n: int
        :rtype: int
        """
        left = 1
        right = n
        while left < right:
            mid = (left + right) >> 1
            if isBadVersion(mid):
                right = mid
            else:
                left = mid + 1
        return left


# 279. Perfect Squares
class Solution(object):
    def numSquares(self, n):
        """
        :type n: int
        :rtype: int
        """
        res = [i for i in range(n + 1)]
        for i in range(2, n + 1):
            j = 2
            while j * j <= i:
                res[i] = min(res[i], res[i - j * j] + 1)
                j += 1
        return res[-1]


if __name__ == "__main__":
    solu = Solution()
    res = solu.numSquares(12)
    print(res)
