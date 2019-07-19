# 21. Merge Two Sorted Lists
# Definition for singly-linked list.


class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution(object):
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        head = ListNode(0)
        tmp = head
        if l1 is None and l2 is None:
            return None
        while l1 is not None or l2 is not None:
            if l1 is None:
                while l2 is not None:
                    tmp.val = l2.val
                    l2 = l2.next
                    if l2 is None:
                        break
                    tmp.next = ListNode(0)
                    tmp = tmp.next
                break
            if l2 is None:
                while l1 is not None:
                    tmp.val = l1.val
                    l1 = l1.next
                    if l1 is None:
                        break
                    tmp.next = ListNode(0)
                    tmp = tmp.next
                break
            if l1.val <= l2.val:
                tmp.val = l1.val
                l1 = l1.next
            else:
                tmp.val = l2.val
                l2 = l2.next
            tmp.next = ListNode(0)
            tmp = tmp.next
        return head
        # test
        # solu = Solution()
        # a = ListNode(1)
        # b = ListNode(2)
        # c = ListNode(3)
        # d = ListNode(4)
        # e = ListNode(5)
        # a.next = c
        # b.next = d
        # c.next = e
        # d.next = None
        # e.next = None
        # ans = solu.mergeTwoLists(a, b)
        # print ans


# 22. Generate Parentheses


class Solution(object):
    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        result = []
        if n == 0:
            return ['']
        if n == 1:
            return ["()"]
        for i in range(n):
            for left in self.generateParenthesis(i):
                for right in self.generateParenthesis(n - i - 1):
                    result.append("({}){}".format(left, right))
                    print i, n - i - 1
                    print "({}){}".format(left, right)
        return result


# 23. Merge k Sorted Lists


class Solution(object):
    def mergeKLists(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        len_n = len(lists)
        if len_n == 0:
            return None
        elif len_n == 1:
            return lists[0]
        n = len_n / 2
        left = self.mergeKLists(lists[:n])
        right = self.mergeKLists(lists[n:])
        return self.mergeTwoLists(left, right)


# 24. Swap Nodes in Pairs


class Solution(object):
    def swapPairs(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if head == None:
            return None
        result = ListNode(0)
        result.next = head
        tmp = result
        while tmp.next and tmp.next.next:
            t = tmp.next.next
            tmp.next.next = t.next
            t.next = tmp.next
            tmp.next = t
            tmp = tmp.next.next
        return result.next


# 26. Remove Duplicates from Sorted Array


class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        len_n = len(nums)
        if len_n == 0:
            return 0
        res = 1
        for i in range(1, len_n, 1):
            if nums[i] != nums[i - 1]:
                nums[res] = nums[i]
                res += 1
        print nums
        return res


# 25. Reverse Nodes in k-Group


class Solution(object):
    def reverseKGroup(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        if head is None or k == 1:
            return head
        count_num = 0
        result = ListNode(0)
        result.next = head
        pre = result
        cur = result
        nex = result
        while cur.next is not None:
            count_num += 1
            cur = cur.next
        while count_num >= k:
            cur = pre.next
            nex = cur.next
            for i in range(1, k, 1):
                cur.next = nex.next
                nex.next = pre.next
                pre.next = nex
                nex = cur.next
            pre = cur
            count_num -= k
        return result.next


# 27. Remove Element


class Solution(object):
    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        count_num = 0
        for i in range(len(nums)):
            if nums[i] != val:
                nums[count_num] = nums[i]
                count_num += 1
        return count_num
        # test = [3,2,2,3]


# 28 Implement strStr()


class Solution(object):
    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        if len(needle) == 0:
            return 0
        if len(needle) > len(haystack):
            return -1
        return haystack.find(needle)


# 29 divide two integers


class Solution(object):
    def divide(self, dividend, divisor):
        flag = True
        if dividend > 0 and divisor < 0:
            flag = False
        if dividend < 0 and divisor > 0:
            flag = False
        res = 0
        dividend = abs(dividend)
        divisor = abs(divisor)
        while divisor <= dividend:
            tmp = divisor
            tmp_res = 1
            while tmp << 1 <= dividend:
                tmp = tmp << 1
                tmp_res = tmp_res << 1
            res += tmp_res
            dividend -= tmp
        if flag:
            if res > 2147483647:
                return 2147483647
            return res
        if res > 2147483648:
            return -2147483648
        return -res


# 30. Substring with Concatenation of All Words


class Solution(object):
    def findSubstring(self, s, words):
        """
        :type s: str
        :type words: List[str]
        :rtype: List[int]
        """
        result = []
        len_s = len(s)
        num_words = len(words)
        if num_words == 0:
            return []
        len_word = len(words[0])
        if num_words * len_word > len_s:
            return result
        words_dict = {}
        for word in words:
            if word not in words_dict:
                words_dict[word] = 1
            else:
                words_dict[word] += 1
        j = 0
        while j <= len_s - num_words * len_word:
            tmp_dict = words_dict.copy()
            i = 0
            while i <= len(s[j:j + num_words * len_word]):
                tmp_word = s[j:j + num_words * len_word][i:i + len_word]
                if tmp_word in tmp_dict and tmp_dict[tmp_word] != 0:
                    tmp_dict[tmp_word] -= 1
                    i += len_word
                else:
                    break
            if i == num_words * len_word:
                result.append(j)
            j += 1
        return result
        #     test = ["wordgoodgoodgoodbestword", ["word","good","best","good"]]


# 31 Next Permutation


class Solution(object):
    def nextPermutation(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        len_nums = len(nums)
        if len_nums <= 1:
            return
        left = len_nums - 2
        while left >= 0:
            if nums[left] < nums[left + 1]:
                right = left + 1
                while right < len_nums:
                    if nums[left] >= nums[right]:
                        break
                    right += 1
                right -= 1
                nums[left], nums[right] = nums[right], nums[left]
                nums[left + 1:] = sorted(nums[left + 1:])
                return
            left -= 1
        nums.reverse()
        return
        #    test = [1, 5, 1]


# 32. Longest Valid Parentheses


class Solution(object):
    def longestValidParentheses(self, s):
        """
        :type s: str
        :rtype: int
        """
        len_s = len(s)
        if len_s < 2:
            return 0
        left = 0
        right = 0
        max_length = 0
        for i in range(len_s):
            if s[i] == '(':
                left += 1
            elif s[i] == ')':
                right += 1
            if left == right:
                max_length = max(left + right, max_length)
            if right > left:
                left = 0
                right = 0
        i = len_s - 1
        left = 0
        right = 0
        while i >= 0:
            if s[i] == '(':
                left += 1
            elif s[i] == ')':
                right += 1
            if left == right:
                max_length = max(left + right, max_length)
            if left > right:
                left = 0
                right = 0
            i -= 1
        return max_length


# 33. search in rotated sorted array


class Solution(object):
    def search(self, nums, target):
        left = 0
        right = len(nums)
        while left < right:
            mid = (left + right) / 2
            if target == nums[mid]:
                return mid
            if nums[left] > nums[mid]:
                if target > nums[mid] and target <= nums[right - 1]:
                    left = mid + 1
                else:
                    right = mid
            else:
                if target >= nums[left] and target < nums[mid]:
                    right = mid
                else:
                    left = mid + 1
        return -1


# 34. Find First and Last Position of Element in Sorted Array


class Solution(object):
    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        left = 0
        right = len(nums) - 1
        result = [-1, -1]
        if right == 0:
            return result
        while left <= right:
            mid = (left + right) / 2
            if nums[mid] == target:
                result[0] = mid
                result[1] = mid
                while mid - 1 >= 0:
                    if nums[mid - 1] != nums[mid]:
                        break
                    mid = mid - 1
                    result[0] = mid
                while mid + 1 < len(nums):
                    if nums[mid + 1] != nums[mid]:
                        break
                    mid = mid + 1
                    result[1] = mid
                return result
            if nums[mid] < target:
                left = mid + 1
                while mid + 1 < len(nums):
                    if nums[mid + 1] != nums[mid]:
                        left = mid + 1
                        break
                    mid = mid + 1
            else:
                right = mid - 1
                while mid - 1 >= 0:
                    if nums[mid - 1] != nums[mid]:
                        right = mid - 1
                        break
                    mid = mid - 1
        return result
        # test =[2] 2

# 35 Search insert rpsition


class Solution(object):
    def searchInsert(self, nums, target):
        left = 0
        right = len(nums) - 1
        if right < 0:
            return 1
        while left < right:
            mid = (left + right) / 2
            if nums[mid] == target:
                return mid
            if nums[mid] < target:
                left = mid + 1
            if nums[mid] > target:
                right = mid - 1
        if nums[left] >= target:
            return left
        else:
            return left + 1
#    test = [2]

#  36. Valid Sudoku


class Solution(object):
    def isValidSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: bool
        """
        for row in range(9):
            tmp_1 = []
            tmp_2 = []
            for col in range(9):
                if board[row][col] != ".":
                    if board[row][col] in tmp_1:
                        return False
                    else:
                        tmp_1.append(board[row][col])
                if board[col][row] != ".":
                    if board[col][row] in tmp_2:
                        return False
                    else:
                        tmp_2.append(board[col][row])
        for row in range(0, 9, 3):
            for col in range(0, 9, 3):
                tmp = []
                for i in range(3):
                    for j in range(3):
                        if board[row + i][col + j] != ".":
                            if board[row + i][col + j] in tmp:
                                return False
                            else:
                                tmp.append(board[row + i][col + j])
        return True


'''
    test = [
        ["5", "3", ".", ".", "7", ".", ".", ".", "."],
        ["6", ".", ".", "1", "9", "5", ".", ".", "."],
        [".", "9", "8", ".", ".", ".", ".", "6", "."],
        ["8", ".", ".", ".", "6", ".", ".", ".", "3"],
        ["4", ".", ".", "8", ".", "3", ".", ".", "1"],
        ["7", ".", ".", ".", "2", ".", ".", ".", "6"],
        [".", "6", ".", ".", ".", ".", "2", "8", "."],
        [".", ".", ".", "4", "1", "9", ".", ".", "5"],
        [".", ".", ".", ".", "8", ".", ".", "7", "9"]
    ]
'''
# 37. Sudoku Solver


class Solution(object):
    def solveSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: None Do not return anything, modify board in-place instead.
        """
        def isvalid(board, x, y):
            for i in range(9):
                if i != x and board[i][y] == board[x][y]:
                    return False
            for j in range(9):
                if j != y and board[x][j] == board[x][y]:
                    return False
            row_x = (x / 3) * 3
            col_y = (y / 3) * 3
            for i in range(3):
                for j in range(3):
                    if (((row_x + i) != x or (col_y + j) != y) and
                            board[row_x + i][col_y + j] == board[x][y]):
                        return False
            return True

        def fillboard(board):
            for row in range(9):
                for col in range(9):
                    if board[row][col] == ".":
                        for item in "123456789":
                            board[row][col] = item
                            if isvalid(board, row, col) and fillboard(board):
                                return True
                            board[row][col] = "."
                        return False
            return True
        fillboard(board)

# 38. Count and Say


class Solution(object):
    def countAndSay(self, n):
        """
        :type n: int
        :rtype: str
        """
        def countstr(tmp_res):
            res = ''
            count = 0
            item = tmp_res[0]
            for i in range(len(tmp_res)):
                if item == tmp_res[i]:
                    count += 1
                else:
                    res += str(count) + item
                    count = 1
                    item = tmp_res[i]
            res += str(count) + item
            return res
        res = '1'
        for i in range(n - 1):
            res = countstr(res)
        return res
        # test=4

# 39. Combination Sum


class Solution(object):
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        def comsum(candidates, target, j):
            res = []
            len_can = len(candidates)
            if target == 0:
                return []
            if target < 0 or j > len_can - 1:
                return [[-1]]
            tmp_no = comsum(candidates, target, j + 1)
            tmp_yes = comsum(candidates, target - candidates[j], j)
            if len(tmp_yes) == 0:
                res.append([candidates[j]])
            elif tmp_yes != [[-1]]:
                for item in tmp_yes:
                    res.append([candidates[j]] + item)
            if tmp_no != [[-1]] and len(tmp_no) != 0:
                for item in tmp_no:
                    res.append(item)
            if tmp_no == [[-1]] and tmp_yes == [[-1]]:
                return [[-1]]
            return res
        candidates.sort()
        result = comsum(candidates, target, 0)
        if result == [[-1]]:
            result = []
        return result
        # test = [2, 3, 5] 8

# 40. Combination Sum II


class Solution(object):
    def combinationSum2(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        def comsum(candidates, target, j):
            res = []
            len_can = len(candidates)
            if target == 0:
                return []
            if target < 0 or j > len_can - 1:
                return [[-1]]
            n = 1
            while n + j < len_can:
                if candidates[j] != candidates[j + n]:
                    break
                n += 1
            tmp_no = comsum(candidates, target, j + n)
            tmp_yes = comsum(candidates, target - candidates[j], j + 1)
            if len(tmp_yes) == 0:
                res.append([candidates[j]])
            elif tmp_yes != [[-1]]:
                for item in tmp_yes:
                    res.append([candidates[j]] + item)
            if tmp_no != [[-1]] and len(tmp_no) != 0:
                for item in tmp_no:
                    res.append(item)
            if tmp_no == [[-1]] and tmp_yes == [[-1]]:
                return [[-1]]
            return res
        candidates.sort()
        result = comsum(candidates, target, 0)
        if result == [[-1]]:
            result = []
        return result

if __name__ == '__main__':
    def shownode(tmp):
        while tmp is not None:
            print test.val
            test = test.next

    solu = Solution()
    test = [10,1,2,7,6,1,5]
    ans = solu.combinationSum2(test, 8)
    print ans
