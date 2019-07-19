# 1
class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        need_data = {}
        for i in xrange(len(nums)):
            if nums[i] in need_data:
                return [need_data[nums[i]], i]
            else:
                need_data[target - nums[i]] = i
        return 0
        # test example     test = [2, 3, 7, 8, 12]


# 2
# Definition for singly-linked list.


class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        tmp1 = str(l1.val)
        tmp2 = str(l2.val)
        l1 = l1.next
        while l1 is not None:
            tmp1 += str(l1.val)
            l1 = l1.next
        l2 = l2.next
        while l2 is not None:
            tmp2 += str(l2.val)
            l2 = l2.next
        sum = str(int(tmp1[::-1]) + int(tmp2[::-1]))[::-1]
        res = ListNode(0)
        tmp = res
        for item in sum:
            tmp.next = ListNode(item)
            tmp = tmp.next
        return res.next


# 3 Longest Substring Without Repeating Characters
class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        logest_str = ''
        max_len = 0
        for i in range(len(s)):
            if s[i] not in logest_str:
                logest_str += s[i]
                if max_len < len(logest_str):
                    max_len += 1
            else:
                index = logest_str.index(s[i])
                logest_str = logest_str[index + 1:] + s[i]
                if len(logest_str) >= max_len:
                    max_len = len(logest_str)
        print logest_str
        return max_len
        # test example test='pwwkew'


# 4. Median of Two Sorted Arrays
class Solution(object):
    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        len1 = len(nums1)
        len2 = len(nums2)
        if (len1 + len2) % 2 == 1:
            left = right = (len1 + len2) / 2
        else:
            right = (len1 + len2) / 2
            left = right - 1
        index_1 = 0
        index_2 = 0
        for i in range(right + 1):
            if index_1 < len1 and index_2 < len2:
                if nums1[index_1] > nums2[index_2]:
                    answer = nums2[index_2]
                    index_2 += 1
                else:
                    answer = nums1[index_1]
                    index_1 += 1
            elif index_1 < len1:
                answer = nums1[index_1]
                index_1 += 1
            elif index_2 < len2:
                answer = nums2[index_2]
                index_2 += 1
            else:
                break
            if i == left:
                answer_left = answer
            if i == right:
                answer_right = answer
        answer = (float(answer_left) + answer_right) / 2
        return answer
        # test example nums1 = [1, 2] nums2 = [3, 4]


# 5. Longest Palindromic Substring

class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        if len(s) <= 1:
            return s
        mark = [[0 for i in range(len(s))] for i in range(len(s))]
        start = 0
        longest = 1
        for i in range(len(s)):
            mark[i][i] = 1
            if i + 1 < len(s):
                if s[i] == s[i + 1]:
                    mark[i][i + 1] = 1
                    start = i
                    longest = 2
        for leng in range(3, len(s) + 1, 1):
            for i in range(len(s)):
                j = i + leng - 1
                if j < len(s):
                    if s[i] == s[j] and mark[i + 1][j - 1] == 1:
                        mark[i][j] = 1
                        start = i
                        longest = leng
        for i in range(len(s)):
            print mark[i]
        return s[start:start + longest]
        # test example 'cbbc'


# 6 ZigZag Conversion

class Solution(object):
    def convert(self, s, numRows):
        """
        :type s: str
        :type numRows: int
        :rtype: str
        """
        length = len(s)
        if numRows == 1 or length <= numRows:
            return s
        new_s = ''
        for i in range(numRows):
            j = i
            if j == 0 or j == numRows - 1:
                while j < length:
                    new_s += s[j]
                    j += 2 * (numRows - 1)
            else:
                while j < length:
                    new_s += s[j]
                    j += 2 * (numRows - 1 - i)
                    if j < length:
                        new_s += s[j]
                        j += 2 * i
        return new_s


# 7. Reverse Integer

class Solution(object):
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        if x == 0:
            return 0
        flag = True if x > 0 else False
        str_x = str(x).strip('-')[::-1].strip('0')
        if flag:
            answer = int(str_x)
            if answer > 2 ** 31:
                answer = 0
        else:
            answer = -1 * int(str_x)
            if answer < -2 ** 31:
                answer = 0
        return answer


# 8 string to interger
class Solution(object):
    def myAtoi(self, str):
        import re
        str = str.strip()
        if len(str) == 0:
            return 0
        neg = 1
        if str[0] == '-':
            neg = -1
            str = str[1:]
        elif str[0] == '+':
            str = str[1:]
        flag = re.search(r'\d+', str)
        if flag is None:
            return 0
        start, end = flag.span()
        if start != 0:
            return 0
        else:
            res = neg * int(str[start:end])
            if res > 2 ** 31 - 1:
                return 2 ** 31 - 1
            elif res < -2 ** 31:
                return -2 ** 31
            else:
                return res
                # test '    1231231441878798'


# 9 Palindrome Number


class Solution(object):
    def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """
        if x < 0 or (x % 10 == 0 and x != 0):
            return False
        result = 0
        while result < x:
            result = result * 10 + x % 10
            x = x / 10
        print result, x
        if result == x or x == result / 10:
            return True
        else:
            return False
            # test 202


# 10. Regular Expression Matching


class Solution(object):
    def isMatch(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: bool
        """
        len_s = len(s)
        len_p = len(p)
        judge = [[False for i in range(len_p + 1)] for i in range(len_s + 1)]
        judge[0][0] = True
        for i in range(1, len_p + 1):
            if p[i - 1] == '*' and i >= 2:
                judge[0][i] = judge[0][i - 2]
        for i in range(1, len_s + 1):
            for j in range(1, len_p + 1):
                if p[j - 1] == '.':
                    judge[i][j] = judge[i - 1][j - 1]
                elif p[j - 1] == '*':
                    if judge[i][j - 1] or judge[i][j - 2]:
                        judge[i][j] = True
                    elif p[j - 2] == s[i - 1] or p[j - 2] == '.':
                        judge[i][j] = judge[i - 1][j]
                else:
                    judge[i][j] = judge[i - 1][j - 1] and p[j - 1] == s[i - 1]
        return judge[len_s][len_p]
        # test "mississippi", "mis*is*ip*."


# 11 Container with most water


class Solution(object):
    def maxArea(self, height):
        left = 0
        right = len(height) - 1
        maxarea = 0
        while left < right:
            high = min(height[left], height[right])
            area = (right - left) * high
            maxarea = max(area, maxarea)
            if high == height[left]:
                left += 1
            else:
                right -= 1
        return maxarea
        # test = [1, 8, 6, 2, 5, 4, 8, 3, 7]


# 12 Integer to roman


class Solution(object):
    def intToRoman(self, num):
        number = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
        roman = ["M", "CM", "D", "CD", "C", "XC",
                 "L", "XL", "X", "IX", "V", "IV", "I"]
        result = ''
        for i in range(len(number)):
            quo = num / number[i]
            num = num % number[i]
            result += quo * roman[i]
            if num == 0:
                break
        return result
        # test=1994 MCMXCIV


# 13 roman to Integer


class Solution(object):
    def RomanToint(self, s):
        number = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
        roman = ["M", "CM", "D", "CD", "C", "XC",
                 "L", "XL", "X", "IX", "V", "IV", "I"]
        maxn = 1
        result = 0
        for i in range(len(s) - 1, -1, -1):
            num = number[roman.index(s[i])]
            if num >= maxn:
                result += num
                maxn = num
            else:
                result -= num
        return result
        # test = 'MCMXCIV'


# 14 loggest common prefix


class Solution(object):
    def longestCommonPrefix(self, strs):
        result = ''
        if len(strs) == 0:
            return result
        result = strs[0]
        for i in range(1, len(strs), 1):
            j = 0
            min_len = min(len(result), len(strs[i]))
            while j < min_len:
                if result[j] != strs[i][j]:
                    break
                j += 1
            result = result[:j]
            if j == 0:
                result = ''
                break
        return result
        # test = ['flower', 'flow', 'flight']


# 15 3Sum


class Solution(object):
    def threesum(self, nums):
        result = []
        len_num = len(nums)
        if len_num < 3:
            return result
        nums.sort()
        i = 0
        while i < len_num - 2:
            left = i + 1
            right = len_num - 1
            target = 0 - nums[i]
            while left < right:
                sum_lr = nums[left] + nums[right]
                if sum_lr < target:
                    left += 1
                elif sum_lr > target:
                    right -= 1
                else:
                    result.append([nums[i], nums[left], nums[right]])
                    left += 1
                    right -= 1
                    while left < right:
                        if nums[left] != nums[left - 1]:
                            break
                        left += 1
                print result
            i += 1
            while i < len_num - 2:
                if nums[i] != nums[i - 1]:
                    break
                i += 1
        return result
        # test = [-1, 0, 1, 2, -1, -4]


# 16 3sum closest


class Solution(object):
    def threeSumClosest(self, nums, target):
        len_num = len(nums)
        if len_num < 3:
            return 0
        nums.sort()
        result = nums[0] + nums[1] + nums[2]
        i = 0
        while i < len_num - 2:
            tmp_target = target - nums[i]
            left = i + 1
            right = len_num - 1
            while left < right:
                tmp_sum = nums[left] + nums[right]
                if tmp_sum == tmp_target:
                    return target
                else:
                    if abs(tmp_target - tmp_sum) < abs(target - result):
                        result = tmp_sum + nums[i]
                    if tmp_sum > tmp_target:
                        right -= 1
                        if nums[left] + nums[left + 1] > tmp_target:
                            if abs(nums[left] + nums[left + 1] + nums[i] - target) < abs(target - result):
                                result = nums[left] + nums[left + 1] + nums[i]
                            break
                    else:
                        left += 1
                        if nums[right] + nums[right - 1] < tmp_target:
                            if abs(nums[right] + nums[right - 1] + nums[i] - target) < abs(target - result):
                                result = nums[right] + \
                                    nums[right + 1] + nums[i]
                            break
            i += 1
        return result
        # test = [-1,2,1,-4] 1

# 17 Letter Combinations of a Phone Number


class Solution(object):
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        result = []
        dic = {'0': ' ', '1': '*', '2': 'abc', '3': 'def',
               '4': 'ghi', '5': 'jkl', '6': 'mno',
               '7': 'pqrs', '8': 'tuv', '9': 'wxyz'}
        for char in digits:
            tmp = []
            for item in dic[char]:
                print item
                if result == []:
                    tmp.append(item)
                else:
                    for res in result:
                        tmp.append(res + item)
            result = tmp
            print result
        return result
        #test = '23'

# 18 4Sum


class Solution(object):
    def foursum(self, nums, target):
        len_nums = len(nums)
        result = []
        if len_nums < 4:
            return result
        nums.sort()
        i = 0
        while i < len_nums - 3:
            j = i + 1
            while j < len_nums - 2:
                left = j + 1
                right = len_nums - 1
                tmp_target = target - nums[i] - nums[j]
                while left < right:
                    sum_lr = nums[left] + nums[right]
                    if nums[left] + nums[left + 1] > sum_lr:
                        break
                    if nums[right] + nums[right - 1] < sum_lr:
                        break
                    if sum_lr < tmp_target:
                        left += 1
                    elif sum_lr > tmp_target:
                        right -= 1
                    else:
                        result.append(
                            [nums[i], nums[j], nums[left], nums[right]])
                        left += 1
                        right -= 1
                        while left < right:
                            if nums[left] != nums[left - 1]:
                                break
                            left += 1
                j += 1
                while j < len_nums - 2:
                    if nums[j] != nums[j - 1]:
                        break
                    j += 1
            i += 1
            while i < len_nums - 3:
                if nums[i] != nums[i - 1]:
                    break
                i += 1
        return result
    #test=[1,0,-1,0,-2,2], 0

# 19 Remove Nth Node From End of List


class Solution(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        result = ListNode(0)
        result.next = head
        tmp1 = result
        tmp2 = result
        i = 0
        while i < n:
            tmp1 = tmp1.next
            i += 1
        while tmp1.next != None:
            tmp1 = tmp1.next
            tmp2 = tmp2.next
        tmp2.next = tmp2.next.next
        return result.next
    # test
    # a=ListNode(1);
    # b=ListNode(2);
    # c=ListNode(3);
    # d=ListNode(4);
    # e=ListNode(5);
    # a.next = b
    # b.next=c
    # c.next=d
    # d.next=e
    # e.next=None
    # ans=removeNthFromEn(a,2)
    # print ans

# 20. Valid Parentheses


class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        len_s = len(s)
        a = len_s % 2
        if a != 0:
            return False
        res = []
        for i in s:
            if i == '(' or i == '[' or i == '{':
                res.append(i)
            if i == ')' or i == ']' or i == '}':
                if len(res) == 0:
                    return False
                tmp = res.pop()
                if i == ')' and tmp != '(':
                    return False
                if i == ']' and tmp != '[':
                    return False
                if i == '}' and tmp != '{':
                    return False
        if len(res) == 0:
            return True
        else:
            return False
        #test="()[]{}"


if __name__ == '__main__':
    solu = Solution()
    test = "()[]{}"
    ans = solu.isValid(test)
    print ans
