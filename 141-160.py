# 141. Linked List Cycle
import collections


class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        if head is None:
            return False
        p1 = head
        p2 = head
        while p1 and p2 and p2.next and p1.next and p2.next.next:
            p1 = p1.next
            p2 = p2.next.next
            if p1 == p2:
                return True
        return False

# 142. Linked List Cycle II


class Solution(object):
    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if head is None or head.next is None:
            return None
        slow = head
        fast = head
        flag = False
        while fast and slow and fast.next and slow.next and fast.next.next:
            fast = fast.next.next
            slow = slow.next
            if slow == fast:
                flag = True
                break
        if slow == fast and flag:
            slow = head
            while slow != fast:
                slow = slow.next
                fast = fast.next
            return slow
        return None

        # 143. Reorder List


class Solution(object):
    def reorderList(self, head):
        """
        :type head: ListNode
        :rtype: None Do not return anything, modify head in-place instead.
        """
        if head is None or head.next is None or head.next.next is None:
            return head
        slow = head
        fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        head1 = head
        head2 = slow.next
        slow.next = None

        dummy = ListNode(0)
        dummy.next = head2
        p = head2.next
        head2.next = None
        while p:
            tmp = p
            p = p.next
            tmp.next = dummy.next
            dummy.next = tmp
        head2 = dummy.next
        p1 = head1
        p2 = head2
        while p2:
            tmp1 = p1.next
            tmp2 = p2.next
            p1.next = p2
            p2.next = tmp1
            p1 = tmp1
            p2 = tmp2
        return

# 144. Binary Tree Preorder Traversal


class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution(object):
    def preorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        res = []
        if root is None:
            return res
        tmp = [root]
        while len(tmp) != 0:
            p = tmp.pop()
            res.append(p.val)
            if p.right:
                tmp.append(p.right)
            if p.left:
                tmp.append(p.left)
        return res

# 145. Binary Tree Postorder Traversal


class Solution(object):
    def postorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        res = []
        if root is None:
            return res
        tmp = [root]
        while len(tmp) != 0:
            p = tmp.pop()
            res.append(p.val)
            if p.left:
                tmp.append(p.left)
            if p.right:
                tmp.append(p.right)
        res.reverse()
        return res


# 146. LRU Cache


class LRUCache(object):

    def __init__(self, capacity):
        """
        :type capacity: int
        """
        self.capacity = capacity
        self.length = 0
        self.content = collections.OrderedDict()

    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        if key in self.content:
            value = self.content[key]
            del self.content[key]
            self.content[key] = value
            return value
        else:
            return -1

    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: None
        """
        if key in self.content:
            del self.content[key]
            self.content[key] = value
        else:
            if self.length == self.capacity:
                self.content.popitem(last=False)
                self.length -= 1
            self.content[key] = value
            self.length += 1
# 147. Insertion Sort List


class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution(object):
    def insertionSortList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if head is None:
            return None
        tmp_head = ListNode(0)
        tmp_head.next = head
        curr = tmp_head.next
        while curr.next:
            if curr.next.val >= curr.val:
                curr = curr.next
            else:
                pre = tmp_head
                while pre.next.val < curr.next.val:
                    pre = pre.next
                tmp = curr.next
                curr.next = curr.next.next
                tmp.next = pre.next
                pre.next = tmp
        return tmp_head.next

# 148. Sort List


class Solution(object):
    def sortList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        def mergetwo(head1, head2):
            if head1 is None:
                return head2
            if head2 is None:
                return head1
            result = ListNode(0)
            p = result
            while head1 and head2:
                if head1.val <= head2.val:
                    p.next = head1
                    head1 = head1.next
                else:
                    p.next = head2
                    head2 = head2.next
                p = p.next
            if head1 is None:
                p.next = head2
            if head2 is None:
                p.next = head1
            return result.next

        if head is None or head.next is None:
            return head
        slow = head
        fast = head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
        head1 = head
        head2 = slow.next
        slow.next = None
        head1 = self.sortList(head1)
        head2 = self.sortList(head2)
        head = mergetwo(head1, head2)
        return head

# 149. Max Points on a Line


class Solution(object):
    def maxPoints(self, points):
        """
        :type points: List[List[int]]
        :rtype: int
        """
        def gcd(a, b):
            if b == 0:
                return a
            else:
                return gcd(b, a % b)
        len_p = len(points)
        if len_p < 3:
            return len_p
        result = -1
        for i in range(len_p):
            res_dict = {"inf": 0}
            samepoint = 1
            for j in range(i + 1, len_p, 1):
                if points[i][0] == points[j][0] and points[i][1] == points[j][1]:
                    samepoint += 1
                    continue
                if points[i][0] == points[j][0] and points[i][1] != points[j][1]:
                    res_dict["inf"] += 1
                else:
                    dx = points[j][0] - points[i][0]
                    dy = points[j][1] - points[i][1]
                    d = gcd(dx, dy)
                    key = (dx / d, dy / d)
                    if key in res_dict:
                        res_dict[key] += 1
                    else:
                        res_dict[key] = 1
            result = max(result, max(res_dict.values()) + samepoint)
        return result

# 150. Evaluate Reverse Polish Notation


class Solution(object):
    def evalRPN(self, tokens):
        """
        :type tokens: List[str]
        :rtype: int
        """
        operators = ["+", "-", "*", "/"]
        data = []
        for item in tokens:
            if item not in operators:
                data.append(int(item))
            else:
                data1 = data.pop()
                data2 = data.pop()
                if item == "+":
                    res = data1 + data2
                elif item == "-":
                    res = data2 - data1
                elif item == "*":
                    res = data1 * data2
                else:
                    if data1 * data2 > 0:
                        res = data2 / data1
                    else:
                        res = -((-data2) / data1)
                data.append(res)
        return data.pop()
#test = ["4", "13", "5", "/", "+"]
# 151. Reverse Words in a String


class Solution(object):
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        data = s.split()
        data.reverse()
        return " ".join(data)
# test = "the sky is blue"

# 152. Maximum Product Subarray


class Solution(object):
    def maxProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        length = len(nums)
        res = [[0 for i in range(length)] for j in range(2)]
        res[0][0] = nums[0]
        res[1][0] = nums[0]
        for i in range(1, length, 1):
            res[0][i] = max(res[0][i - 1] * nums[i], res[1]
                            [i - 1] * nums[i], nums[i])
            res[1][i] = min(res[0][i - 1] * nums[i], res[1]
                            [i - 1] * nums[i], nums[i])
        return max(res[0])
        # test = [2, 3, -2, 4]
# 153. Find Minimum in Rotated Sorted Array


class Solution(object):
    def findMin(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        left = 0
        right = len(nums) - 1
        if right <= 0:
            return nums[0]
        while left < right:
            mid = (left + right) / 2
            if nums[mid] < nums[right]:
                right = mid
            else:
                left = mid + 1
        return nums[left]
        # test=[3,4,5,1,2]


# 154. Find Minimum in Rotated Sorted Array II
class Solution(object):
    def findMin(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        left = 0
        right = len(nums) - 1
        if right <= 0:
            return nums[0]
        while left < right and nums[left] >= nums[right]:
            mid = (left + right) / 2
            if nums[mid] < nums[right]:
                right = mid
            elif nums[mid] > nums[left]:
                left = mid + 1
            else:
                left += 1
        return nums[left]
        # test = [1,3,3]

# 155. Min Stack


class MinStack(object):

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.data = []

    def push(self, x):
        """
        :type x: int
        :rtype: None
        """
        self.data.append(x)

    def pop(self):
        """
        :rtype: None
        """
        res = self.data.pop()
        return res

    def top(self):
        """
        :rtype: int
        """
        res = None
        length = len(self.data)
        if length > 0:
            res = self.data[length - 1]
        return res

    def getMin(self):
        """
        :rtype: int
        """
        res = min(self.data)
        return res

# 160. Intersection of Two Linked Lists


class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        def get_len(head):
            length = 0
            while head:
                head = head.next
                length += 1
            return length
        if headA is None or headB is None:
            return None
        lengthA = get_len(headA)
        lengthB = get_len(headB)
        if lengthA < lengthB:
            tmp = headB
            headB = headA
            headA = tmp
        tmp = abs(lengthA - lengthB)
        while tmp > 0:
            headA = headA.next
            tmp -= 1
        while headA is not None and headB is not None:
            if headA == headB:
                return headA
            else:
                headA = headA.next
                headB = headB.next

        return None


if __name__ == '__main__':
    solu = Solution()
    test = [1, 3, 3]
    res = solu.findMin(test)
    print res
