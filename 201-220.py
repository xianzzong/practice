# -*- coding: utf-8 -*
# 201. Bitwise AND of Numbers Range


class Solution(object):
    def rangeBitwiseAnd(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        i = 0
        while m != n:
            m = m >> 1
            n = n >> 1
            i += 1
        return m << i
# res=rangeBitwiseAnd(10,11)
# 202. Happy Number


class Solution(object):
    def isHappy(self, n):
        """
        :type n: int
        :rtype: bool
        """
        def getdigits(number):
            result = 0
            while number:
                remainer = number % 10
                result += remainer**2
                number = number / 10
            return result
        sum_sq_digits = [n]
        while True:
            tmp_sum = getdigits(sum_sq_digits[-1])
            if tmp_sum == 1:
                return True
            if tmp_sum in sum_sq_digits:
                return False
            sum_sq_digits.append(tmp_sum)
# res = solu.isHappy(19)

# 203. Remove Linked List Elements


class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution(object):
    def removeElements(self, head, val):
        """
        :type head: ListNode
        :type val: int
        :rtype: ListNode
        """
        res = ListNode(0)
        res.next = head
        tmp = res
        while tmp and tmp.next:
            if tmp.next.val == val:
                tmp.next = tmp.next.next
            else:
                tmp = tmp.next
        return res.next
        # res = solu.isHappy(19)
# 204. Count Primes


class Solution(object):
    def countPrimes(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n < 2:
            return 0
        res = [1] * n
        res[0] = 0
        res[1] = 0
        i = 2
        while i**2 < n:
            if res[i]:
                tmp = i**2
                while tmp < n:
                    res[tmp] = 0
                    tmp += i
            i += 1
        return sum(res)
#         res = solu.countPrimes(10)
#
# 205. Isomorphic Strings


class Solution(object):
    def isIsomorphic(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        sourcemap = {}
        targetmap = {}
        length = len(s)
        for i in range(length):
            source = sourcemap.get(t[i])
            target = targetmap.get(s[i])
            if source is None and target is None:
                sourcemap[t[i]] = s[i]
                targetmap[s[i]] = t[i]
            elif source != s[i] or target != t[i]:
                return False
        return True
        # res = solu.isIsomorphic("egg", "add")

# 206. Reverse Linked List


class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution(object):
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        res = ListNode(0)
        while head:
            tmp = head.next
            head.next = res.next
            res.next = head
            head = tmp
        return res.next

# 207. Course Schedule


class Solution(object):
    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
# degree[i]用来记录课程i的先导课程数目
# childs[i]代表课程i是其中元素代表课程的先导课程

        degree = [0] * numCourses
        childs = [[] for i in range(numCourses)]
        for item in prerequisites:
            degree[item[0]] += 1
            childs[item[1]].append(item[0])
        courses = set(range(numCourses))
        flag = True
        while flag and len(courses):
            flag = False
            removelist = []
            for x in courses:
                if degree[x] == 0:
                    for child in childs[x]:
                        degree[child] -= 1
                    removelist.append(x)
                    flag = True
            for item in removelist:
                courses.remove(item)
        return len(courses) == 0
# res = solu.canFinish(2, [[0, 1], [1, 0]])


# 208. Implement Trie(Prefix Tree)
class Trie(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.result = []

    def insert(self, word):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: void
        """
        self.result.append(word)

    def search(self, word):
        """
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        return word in self.result

    def startsWith(self, prefix):
        """
        Returns if there is any word in the trie that starts with the given prefix.
        :type prefix: str
        :rtype: bool
        """
        for item in self.result:
            flag = item.startswith(prefix)
            if flag:
                return True
        return False

# 209. Minimum Size Subarray Sum


class Solution(object):
    def minSubArrayLen(self, s, nums):
        """
        :type s: int
        :type nums: List[int]
        :rtype: int
        """
        if sum(nums) < s:
            return 0
        length = len(nums)
        min_len = length + 1
        start = 0
        end = 0
        sum_window = 0
        while end < length:
            while end < length and sum_window < s:
                sum_window += nums[end]
                end += 1
            while start < length and sum_window >= s:
                min_len = min(min_len, end - start)
                sum_window -= nums[start]
                start += 1
        return min_len
        # res = solu.minSubArrayLen(7, [2, 3, 1, 2, 4, 3])
# 210. Course Schedule II


class Solution(object):
    def findOrder(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: List[int]
        """
        degree = [0] * numCourses
        childs = [[] for i in range(numCourses)]
        for item in prerequisites:
            degree[item[0]] += 1
            childs[item[1]].append(item[0])

        tmp = []
        for i in range(numCourses):
            if degree[i] == 0:
                tmp.append(i)

        res = []
        while tmp:
            course = tmp.pop()
            res.append(course)
            for child in childs[course]:
                degree[child] -= 1
                if degree[child] == 0:
                    tmp.append(child)
        if len(res) == numCourses:
            return res
        else:
            return []
# 211. Add and Search Word - Data structure design


class Solution(object):
    def findOrder(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: List[int]
        """
        degree = [0] * numCourses
        childs = [[] for i in range(numCourses)]
        for item in prerequisites:
            degree[item[0]] += 1
            childs[item[1]].append(item[0])

        tmp = []
        for i in range(numCourses):
            if degree[i] == 0:
                tmp.append(i)

        res = []
        while tmp:
            print tmp
            course = tmp.pop()
            res.append(course)
            for child in childs[course]:
                degree[child] -= 1
                if degree[child] == 0:
                    tmp.append(child)
        if len(res) == numCourses:
            return res
        else:
            return []


class TrieNode:
    def __init__(self):
        self.children = {}
        self.isword = False


class WordDictionary(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = TrieNode()

    def addWord(self, word):
        """
        Adds a word into the data structure.
        :type word: str
        :rtype: None
        """
        node = self.root
        for letter in word:
            child = node.children.get(letter)
            if child is None:
                child = TrieNode()
                node.children[letter] = child
            node = child
        node.isword = True

    def search(self, word):
        """
        Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter.
        :type word: str
        :rtype: bool
        """
        return self.find(word, self.root)

    def find(self, word, node):
        if len(word) == 0:
            return node.isword
        if word[0] == ".":
            for key, item in node.children.iteritems():
                if self.find(word[1:], item):
                    return True
        else:
            child = node.children.get(word[0])
            if child is None:
                return False
            else:
                return self.find(word[1:], child)

# 212. Word Search II


class TreeNode(object):
    def __init__(self):
        self.children = {}
        self.isword = False


class Tree(object):
    def __init__(self):
        self.root = TreeNode()

    def insert(self, word):
        node = self.root
        for letter in word:
            child = node.children.get(letter)
            if child is None:
                child = TreeNode()
                node.children[letter] = child
            node = child
        node.isword = True

    def delete(self, word):
        need_delete = []
        node = self.root
        for letter in word:
            need_delete.append((letter, node))
            child = node.children.get(letter)
            if child is None:
                return False
            node = child
        if not node.isword:
            return False
        if len(node.children):
            node.isword = False
        else:
            for letter, node in reversed(need_delete):
                del node.children[letter]
                if node.children or node.isword:
                    break
        return True


class Solution(object):
    def findWords(self, board, words):
        """
        :type board: List[List[str]]
        :type words: List[str]
        :rtype: List[str]
        """
        res = []
        if len(board) == 0:
            return res
        row = len(board)
        col = len(board[0])
        visited = [[False for x in range(col)] for y in range(row)]
        tree = Tree()
        for word in words:
            tree.insert(word)

        def search(word, node, x, y):
            move = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            child = node.children.get(board[x][y])
            if child is None:
                return
            node = child
            visited[x][y] = True

            for item_x, item_y in move:
                new_x = x + item_x
                new_y = y + item_y
                if (new_x >= 0 and new_y >= 0 and new_x < row and new_y < col
                        and not visited[new_x][new_y]):
                    search(word + board[new_x][new_y], node, new_x, new_y)
            visited[x][y] = False
            if node.isword:
                res.append(word)
                tree.delete(word)
            return res
        for x in range(row):
            for y in range(col):
                search(board[x][y], tree.root, x, y)
        return res

# 213. House Robber II


class Solution(object):
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        length = len(nums)
        if length == 0:
            return 0
        elif length == 1:
            return nums[0]

        def get_max_money(homes):
            if len(homes) == 0:
                return 0
            elif len(homes) == 1:
                return homes[0]
            result = [0 for i in range(len(homes))]
            result[0] = homes[0]
            result[1] = max(result[0], homes[1])
            for i in range(2, len(homes)):
                result[i] = max(result[i - 2] + homes[i], result[i - 1])
            return result[-1]
        tmp1 = get_max_money(nums[:-1])
        tmp2 = get_max_money(nums[1:])
        return max(tmp1, tmp2)
        # res = solu.rob([2, 3, 2])
# 214. Shortest Palindrome


class Solution(object):
    def shortestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        l = s + "*" + s[::-1]
        length = len(l)
        p = [0 for i in range(length)]
        for i in range(1, length, 1):
            j = p[i - 1]
            while j > 0 and l[i] != l[j]:
                j = p[j - 1]
            p[i] = j + (l[i] == l[j])
        res = s[::-1][:len(s) - p[-1]] + s
        return res
# res = solu.shortestPalindrome("aacecaaa")
# 215. Kth Largest Element in an Array


class Solution(object):
    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        def partition(nums, left, right):
            pivot = nums[left]
            l = left + 1
            r = right
            while l <= r:
                if nums[l] < pivot and nums[r] > pivot:
                    nums[l], nums[r] = nums[r], nums[l]
                    l += 1
                    r -= 1
                if nums[l] >= pivot:
                    l += 1
                if nums[r] <= pivot:
                    r -= 1
            nums[left], nums[r] = nums[r], nums[left]
            return r
        left = 0
        right = len(nums) - 1
        while True:
            pos = partition(nums, left, right)
            print nums, pos
            if pos == k - 1:
                return nums[pos]
            if pos > k - 1:
                right = pos - 1
            else:
                left = pos + 1

        return partition(nums, 0, len(nums) - 1)

# 217. Contains Duplicate


class Solution(object):
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        data = set(nums)
        return len(data) != len(nums)

# 219. Contains Duplicate II


class Solution(object):
    def containsNearbyDuplicate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        """
        res_dict = {}
        flag = False
        for i in range(len(nums)):
            if nums[i] in res_dict:
                if abs(i - res_dict[nums[i]]) <= k:
                    flag = True
                    break
                else:
                    res_dict[nums[i]] = i
            else:
                res_dict[nums[i]] = i
        return flag

# 220. Contains Duplicate III


class Solution(object):
    def containsNearbyAlmostDuplicate(self, nums, k, t):
        """
        :type nums: List[int]
        :type k: int
        :type t: int
        :rtype: bool
        """
        res_dict = {}
        if k <= 0 or t < 0:
            return False
        for i, num in enumerate(nums):
            key = num // (t + 1)
            if key in res_dict \
                    or (key + 1 in res_dict and res_dict[key + 1] - num <= t)\
                    or (key - 1 in res_dict and num - res_dict[key - 1] <= t):
                return True
            if i >= k:
                del res_dict[nums[i - k] // (t + 1)]
            res_dict[key] = num
        return False

# 216. Combination Sum III


class Solution(object):
    def combinationSum3(self, k, n):
        """
        :type k: int
        :type n: int
        :rtype: List[List[int]]
        """
        res = []

        def dfs(nums, numbers, sums, index, path):
            if sums < 0 or numbers < 0:
                return
            if sums == 0 and numbers == 0:
                res.append(path)
                return
            for i in range(index, len(nums)):
                dfs(nums, numbers - 1, sums - nums[i], i + 1, path + [nums[i]])
            return
        data = range(1, 10)
        dfs(data, k, n, 0, [])
        return res
        # res = solu.combinationSum3(3, 9)
# 218. The Skyline Problem


class Solution(object):
    def getSkyline(self, buildings):
        from heapq import heappush, heappop
        # 对于一个 building, 他由 (l, r, h) 三元组组成, 我们可以将其分解为两种事件:
        #     1. 在 left position, 高度从 0 增加到 h(并且这个高度将持续到 right position);
        #     2. 在 right position, 高度从 h 降低到 0.
        # 由此引出了 event 的结构: 在某一个 position p, 它引入了一个高度为 h 的 skyline, 将一直持续到另一 end postion

        # 对于在 right position 高度降为 0 的 event, 它的持续长度是无效的
        # 只保留一个 right position event, 就可以同时触发不同的两个 building
        # 在同一 right position 从各自的 h 降为 0 的 event, 所以对 right position events 做集合操作会减少计算量

        # 由于需要从左到右触发 event, 所以按 postion 对 events 进行排序
        # 并且, 对于同一 positon, 我们需要先触发更高 h 的事件, 先触发更高 h 的事件后,
        # 那么高的 h 相比于低的 h 会占据更高的 skyline, 低 h 的 `key point` 就一定不会产生; 相反, 可能会从低到高连续产生冗余的 `key point`
        # 所以, event 不仅需要按第一个元素 position 排序, 在 position 相同时, 第二个元素 h 也是必须有序的
        events = sorted([(l, -h, r) for l, r, h in buildings] +
                        list({(r, 0, 0) for l, r, h in buildings}))
        print events

        # res 记录了 `key point` 的结果: [x, h]
        # 同时 res[-1] 处的 `key point` 代表了在下一个 event 触发之前, 一直保持的最高的 skyline

        # hp 记录了对于一条高为 h 的 skyline, 他将持续到什么 position 才结束: [h, endposition]
        # 在同时有多条 skyline 的时候, h 最高的那条 skyline 会掩盖 h 低的 skyline, 因此在 event 触发时, 需要得到当前最高的 skyline
        # 所以利用 heap 结构存储 hp, 它的第一个值永远为列表中的最小值: 因此在 event 中记录的是 -h, heap 结构就会返回最高的 skyline. 同时, h 必须在 endposition 之前, 因为它按第一个元素排序
        res, hp = [[0, 0]], [(0, float('inf'))]

        for l, neg_h, r in events:
            # 触发 event 时, 首先要做的就是清除已经到 endposition 的 skyline
            # hp: [h, endposition]
            # 如果当前 position 大于等于了 hp 中的 endposition, 那么该 skyline 会被清除掉
            # 由于在有 high skyline 的情况下, low skyline 不会有影响, 因此, 只需要按从高到低的方式清除 skyline, 直到剩下一个最高的 skyline 并且它的 endposition 大于当前 position
            while l >= hp[0][1]:
                heappop(hp)

            # 对于高度增加到 h 的时间(neg_h < 0), 我们需要添加一个 skyline, 他将持续到 r 即 endposition
            if neg_h:
                heappush(hp, (neg_h, r))

            # 由于 res[-1][1] 记录了在当前事件触发之前一直保持的 skyline
            # 如果当前事件触发后 skyline 发生了改变
            #     1. 来了一条新的高度大于 h 的 skyline
            #     2. res[-1] 中记录的 skyline 到达了 endposition
            # 这两种事件都会导致刚才持续的 skyline 与现在最高的 skyline 不同; 同时, `key point` 产生了, 他将被记录在 res 中
            if res[-1][1] != -hp[0][0]:
                res.append([l, -hp[0][0]])

        return res[1:]

    def getSkyline2(self, buildings):
        from heapq import heappush, heappop
        events = set()
        for l, r, h in buildings:
            events.add((l, -h, r))
            events.add((r, 0, 0))
        events = list(events)
        events = sorted(events)
        print events
        res = [[0, 0]]
        hp = [(0, float("inf"))]
        for l, h, r in events:
            while l >= hp[0][1]:
                heappop(hp)
            if h:
                heappush(hp, (h, r))
            if res[-1][1] != -hp[0][0]:
                res.append([l, -hp[0][0]])
        return res[1:]


if __name__ == "__main__":
    solu = Solution()
    res = solu.getSkyline2(
        [[2, 9, 10], [3, 7, 15], [5, 12, 12], [15, 20, 10], [19, 24, 8]])
    print res
