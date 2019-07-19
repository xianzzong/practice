# 81. Search in Rotated Sorted Array II
class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: bool
        """
        first = 0
        last = len(nums) - 1
        while first <= last:
            mid = (first + last) / 2
            if nums[mid] == target:
                return True
            if nums[first] == nums[mid] == nums[last]:
                first += 1
                last -= 1
            elif nums[first] <= nums[mid]:
                if nums[mid] > target and nums[first] <= target:
                    last = mid - 1
                else:
                    first = mid + 1
            else:
                if nums[mid] < target and nums[last] >= target:
                    first = mid + 1
                else:
                    last = mid - 1
        return False
        #test = [1,3,1,1,1]


# 83. Remove Duplicates from Sorted List
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution(object):
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if head is None:
            return head
        tmp = head
        while tmp.next:
            if tmp.val == tmp.next.val:
                tmp.next = tmp.next.next
            else:
                tmp = tmp.next
        return head


class Solution(object):
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        res = ListNode(0)
        res.next = head
        recent = res
        tmp = res.next
        while recent.next:
            while tmp.next and tmp.next.val == recent.next.val:
                tmp = tmp.next
            if tmp == recent.next:
                recent = recent.next
                tmp = recent.next
            else:
                recent.next = tmp.next
        return res.next

    # a, b, c, d, e = ListNode(1), ListNode(
    #     1), ListNode(1), ListNode(2), ListNode(3)
    # a.next = b
    # b.next = c
    # c.next = d
    # d.next = e
# 84. Largest Rectangle in Histogram


class Solution(object):
    def largestRectangleArea(self, heights):
        """
        :type heights: List[int]
        :rtype: int
        """
        max_area = 0
        height_list = []
        index_list = []
        for i in range(len(heights)):
            if len(height_list) == 0 or heights[i] >= height_list[len(height_list) - 1]:
                height_list.append(heights[i])
                index_list.append(i)
            elif heights[i] < height_list[len(height_list) - 1]:
                while len(height_list) > 0 and heights[i] < height_list[len(height_list) - 1]:
                    height = height_list.pop()
                    last_index = index_list.pop()
                    max_area = max(max_area, height * (i - last_index))
                if len(height_list) == 0 or heights[i] >= height_list[len(height_list) - 1]:
                    height_list.append(heights[i])
                    index_list.append(last_index)
        while len(height_list) > 0:
            height = height_list.pop()
            last_index = index_list.pop()
            max_area = max(max_area, height * (len(heights) - last_index))
        return
        #test = [2, 1, 2]

# 85. Maximal Rectangle


class Solution(object):
    def maximalRectangle(self, matrix):
        """
        :type matrix: List[List[str]]
        :rtype: int
        """
        def largestarea(heights):
            max_row = 0
            index_list = []
            i = 0
            while i < len(heights):
                if len(index_list) == 0 or heights[i] > heights[index_list[len(index_list) - 1]]:
                    index_list.append(i)
                else:
                    current = index_list.pop()
                    if len(index_list) == 0:
                        area = i * heights[current]
                    else:
                        area = heights[current] * \
                            (i - index_list[len(index_list) - 1] - 1)
                    max_row = max(max_row, area)
                    i -= 1
                i += 1
            while len(index_list) > 0:
                current = index_list.pop()
                if len(index_list) == 0:
                    area = i * heights[current]
                else:
                    area = heights[current] * \
                        (i - index_list[len(index_list) - 1] - 1)
                max_row = max(max_row, area)
            return max_row
        max_area = 0
        row = len(matrix)
        if row == 0:
            return 0
        col = len(matrix[0])
        if col == 0:
            return 0
        res = [0 for i in range(col)]
        for i in range(row):
            for j in range(col):
                if matrix[i][j] == "0":
                    res[j] = 0
                else:
                    res[j] += 1
            area = largestarea(res)
            max_area = max(area, max_area)
        return max_area
        # test = [
        #     ["1", "0", "1", "0", "0"],
        #     ["1", "0", "1", "1", "1"],
        #     ["1", "1", "1", "1", "1"],
        #     ["1", "0", "0", "1", "0"]
        # ]

# 86. Partition List


class Solution(object):
    def partition(self, head, x):
        """
        :type head: ListNode
        :type x: int
        :rtype: ListNode
        """
        first = ListNode(0)
        last = ListNode(0)
        tmp_first = first
        tmp_last = last
        while head:
            if head.val < x:
                first.next = head
                head = head.next
                first = first.next
                first.next = None
            else:
                last.next = head
                head = head.next
                last = last.next
                last.next = None
        first.next = tmp_last.next
        return tmp_first.next

# 87. Scramble String


class Solution(object):
    def isScramble(self, s1, s2):
        """
        :type s1: str
        :type s2: str
        :rtype: bool
        """
        def judger(string1, string2):
            if string1 == string2:
                return True
            if len(string1) != len(string2):
                return False
            list1 = list(string1)
            list1.sort()
            list2 = list(string2)
            list2.sort()
            if list1 != list2:
                return False
            for i in range(1, len(string1), 1):
                if judger(string1[:i], string2[:i]) and judger(string1[i:], string2[i:]):
                    return True
                elif judger(string1[:i], string2[len(string2) - i:]) and judger(string1[i:], string2[:len(string1) - i]):
                    return True
            return False
        return judger(s1, s2)
# 88. Merge Sorted Array


class Solution(object):
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: None Do not return anything, modify nums1 in-place instead.
        """
        index_1 = m - 1
        index_2 = n - 1
        index_res = m + n - 1
        while index_1 >= 0 and index_2 >= 0:
            if nums1[index_1] > nums2[index_2]:
                nums1[index_res] = nums1[index_1]
                index_1 -= 1
            else:
                nums1[index_res] = nums2[index_2]
                index_2 -= 1
            index_res -= 1
        while index_2 >= 0:
            nums1[index_res] = nums2[index_2]
            index_2 -= 1
            index_res -= 1
        return
        # test = [[1,2,3,0,0,0], [2,5,6]]

# 89. Gray Code


class Solution(object):
    def grayCode(self, n):
        """
        :type n: int
        :rtype: List[int]
        """
        res = [0]
        for i in range(n):
            j = len(res) - 1
            while j >= 0:
                res.append(res[j] | (1 << i))
                j -= 1
        return res
        #test = 2

# 90. Subsets II


class Solution(object):
    def subsetsWithDup(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        def solve(numbers):
            res = []
            length = len(numbers)
            if length == 0:
                return [[]]
            numbers.sort()
            tmp = solve(numbers[:length - 1])
            for item in tmp:
                if item not in res:
                    res.append(item)
                tmp_item = item + [numbers[length - 1]]
                if tmp_item not in res:
                    res.append(tmp_item)
            return res
        result = solve(nums)
        return result
        # test = [1,2,2]
# 91. Decode Ways


class Solution(object):
    def numDecodings(self, s):
        """
        :type s: str
        :rtype: int
        """
        len_s = len(s)
        if len_s == 0 or s[0] == "0":
            return 0
        res = [1, 1]
        for i in range(2, len_s + 1, 1):
            tmp = int(s[i - 2:i])
            if 10 < tmp <= 26 and tmp != 20:
                res.append(res[i - 2] + res[i - 1])
            elif tmp == 10 or tmp == 20:
                res.append(res[i - 2])
            elif s[i - 1] != "0":
                res.append(res[i - 1])
            else:
                return 0
        return res[len_s]

# 92. Reverse Linked List II


class Solution(object):
    def reverseBetween(self, head, m, n):
        """
        :type head: ListNode
        :type m: int
        :type n: int
        :rtype: ListNode
        """
        if head is None or head.next is None:
            return head
        res = ListNode(0)
        res.next = head
        t = res
        for i in range(m - 1):
            t = t.next
        tmp = t.next
        for i in range(n - m):
            tmp1 = t.next
            t.next = tmp.next
            tmp.next = tmp.next.next
            t.next.next = tmp1
        return res.next

# 93. Restore IP Addresses


class Solution(object):
    def restoreIpAddresses(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        def solve(stri, n):
            len_s = len(stri)
            if len_s > 3 * n or (len_s > 0 and n <= 0) or (len_s == 0 and n > 0):
                return []
            if n == 1:
                if len_s > 1 and stri[0] == "0":
                    return []
                elif int(stri) <= 255:
                    return [stri]
                else:
                    return []
            i = 1
            res = []
            if stri[0] == "0":
                tmp = solve(stri[1:], n - 1)
                for item in tmp:
                    res.append("0." + item)
            else:
                while i < 4:
                    tmp_str = stri[:i]
                    if int(tmp_str) <= 255:
                        tmp = solve(stri[i:], n - 1)
                        if len(tmp) != 0:
                            for item in tmp:
                                res.append(tmp_str + "." + item)
                    i += 1
            return res
        res = solve(s, 4)
        return res
        # test = "010010"
# 94. Binary Tree Inorder Traversal
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution(object):
    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        def solve(tree):
            if tree is None:
                return []
            return solve(tree.left)+[tree.val]+solve(tree.right)
        return solve(root)

if __name__ == '__main__':
    solu = Solution()
    a,b,c=TreeNode(1),TreeNode(2),TreeNode(3)
    a.right=b
    b.left=c
    res = solu.inorderTraversal(a)
    print res
