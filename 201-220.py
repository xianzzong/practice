# coding=utf-8
def rangeBitwiseAnd(m, n):
    if m == n:
        return m
    bin_n = bin(n)[2:]
    bin_m = bin(m)[2:]
    max_len = len(bin_n)
    min_len = len(bin_m)
    if max_len > min_len:
        return 0
    else:
        res = ''
        for i in range(min_len):
            if bin_m[i] == bin_n[i]:
                res += bin_m[i]
            else:
                break
        res = res + '0' * (min_len - len(res))
    res = int(res, 2)
    return res


# res=rangeBitwiseAnd(10,11)
# print res
def isHappy(n):
    def getdigits(n):
        res = []
        while n:
            res.append(n % 10)
            n = n / 10
        result = sum([x ** 2 for x in res])
        return result

    sum_sq_digits = [n]
    while True:
        tmp = getdigits(sum_sq_digits[-1])
        if tmp == 1:
            return True
        elif tmp in sum_sq_digits:
            return False
        sum_sq_digits.append(tmp)


# res=isHappy(123)
# print res
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


def removeElements(head, val):
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


# a,b,c,d,e,f,g=ListNode(1),ListNode(1),ListNode(6),ListNode(3),ListNode(4),ListNode(5),ListNode(6)
# a.next=b
# b.next=None
# c.next=d
# d.next=e
# e.next=f
# f.next=g
# res=removeElements(a,1)
# print res.val
def countPrimes(n):
    if n < 2:
        return 0
    result = [True] * n
    result[0] = False
    result[1] = False
    i = 2
    while i ** 2 < n:
        if result[i]:
            judge = i ** 2
            while judge < n:
                result[judge] = False
                judge += i
        i += 1
    return sum(result)


# res=countPrimes(20)
# print res
def isIsomorphic(s, t):
    """
    :type s: str
    :type t: str
    :rtype: bool
    """
    # TLE
    # length=len(s)
    # flag=[True]*length
    # for i in range(length):
    #     if (s[i]==t[i]) or(t[i] in t[0:i]):
    #         continue
    #     tmp=s[i]
    #     if flag[i]:
    #         s=s[0:i]+t[i]+s[i+1:]
    #         flag[i]=False
    #     x=0
    #     while tmp in s[x:]:
    #         i_index=s[x:].index(tmp)
    #         i_index=i_index+x
    #         if flag[i_index]:
    #             s = s[0:i_index] + t[i] + s[i_index + 1:]
    #             flag[i_index]=False
    #         x+=1
    # if s==t:
    #     return True
    # else:
    #     return False
    sourcemap = {};
    targetmap = {}
    for i in range(len(s)):
        source, target = sourcemap.get(t[i]), targetmap.get(s[i])
        if source == None and target == None:
            sourcemap[t[i]] = s[i]
            targetmap[s[i]] = t[i]
        elif source != s[i] or target != t[i]:
            return False
    return True


# res=isIsomorphic('baba','abab')
# print res
def reverseList(head):
    # res=ListNode(0)
    # while head:
    #     tmp=head.next
    #     head.next=res.next
    #     res.next=head
    #     head=tmp
    # return res.next
    def doreverse(head, newhead):
        if head == None:
            return newhead
        next = head.next
        head.next = newhead
        return doreverse(next, head)

    return doreverse(head, None)


# a,b,c,d,e,f,g=ListNode(1),ListNode(2),ListNode(3),ListNode(4),ListNode(5),ListNode(6),ListNode(7)
# a.next=b
# b.next=c
# c.next=None
# d.next=e
# e.next=f
# f.next=g
# res=reverseList(a)
# print res
def canFinish(numCourses, prerequisites):
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
                # removelist中代表已经修过的课程
        for item in removelist:
            courses.remove(item)
    return len(courses) == 0


# res=canFinish(13,[[0,2],[3,2],[1,0],[5,0],
#                   [6,0],[4,5],[5,3],[4,6],
#                   [9,6],[6,7],[7,8],[10,9],
#                   [12,9],[11,9],[12,11]])
# print res
class TrieNode:
    def __init__(self):
        self.children = {}
        self.isword = False


class Trie(object):
    # TLE
    # def __init__(self):
    #     """
    #     Initialize your data structure here.
    #     """
    #     self.result=[]
    # def insert(self, word):
    #     """
    #     Inserts a word into the trie.
    #     :type word: str
    #     :rtype: void
    #     """
    #     self.result.append(word)
    # def search(self, word):
    #     """
    #     Returns if the word is in the trie.
    #     :type word: str
    #     :rtype: bool
    #     """
    #     return word in self.result
    # def startsWith(self, prefix):
    #     """
    #     Returns if there is any word in the trie that starts with the given prefix.
    #     :type prefix: str
    #     :rtype: bool
    #     """
    #     for item in self.result:
    #         flag=item.startswith(prefix)
    #         if flag:
    #             return True
    #     return False
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = TrieNode()

    def insert(self, word):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: void
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
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        node = self.root
        for letter in word:
            node = node.children.get(letter)
            if node is None:
                return False
        return node.isword

    def startsWith(self, prefix):
        """
        Returns if there is any word in the trie that starts with the given prefix.
        :type prefix: str
        :rtype: bool
        """
        node = self.root
        for letter in prefix:
            node = node.children.get(letter)
            if node is None:
                return False
        return True


# obj=Trie()
# obj.insert('hello')
# obj.insert('word')
# print obj.startsWith('wo')
# print obj.startsWith('f')
# print obj.search('word')
# print obj.search('my')
def minSubArrayLen(s, nums):
    if sum(nums) < s:
        return 0
    len_n = len(nums)
    min_len = len_n + 1
    start, end, subsum = 0, 0, 0
    while end < len_n:
        while end < len_n and subsum < s:
            subsum += nums[end]
            end += 1
        while start < len_n and subsum >= s:
            min_len = min(min_len, end - start)
            subsum -= nums[start]
            start += 1
    return min_len


# res=minSubArrayLen(7,[2,3,1,2,4,3])
# print res
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
        :rtype: void
        """
        node = self.root
        for letter in word:
            child = node.children.get(letter)
            if child == None:
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
        return self.find(self, self.root, word)

    def find(self, node, word):
        if word == '':
            return node.isword
        if word[0] == '.':
            for x in node.children:
                if self.find(node.children[x], word[1:]):
                    return True
        else:
            child = node.children.get(word[0])
            if child:
                return self.find(child, word[1:])
        return False


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


def findWords(board, words):
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


# test = [
#   ['o','a','a','n'],
#   ['e','t','a','e'],
#   ['i','h','k','r'],
#   ['i','f','l','v']
# ]
# words = ["oath","pea","eat","rain"]
# res=findWords(test, words)
# print (res)

##House Robber II

def rob(nums):
    if len(nums) == 0:
        return 0
    elif len(nums) == 1:
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

    max_money_one = get_max_money(nums[:-1])
    max_money_two = get_max_money(nums[1:])
    return max(max_money_one, max_money_two)


# test = [1,2,3,1]
# res = rob(test)
# print (res)
def shortestPalindrome(s):
    l = s + '#' + s[::-1]
    p = [0 for i in range(len(l))]
    for i in range(1, len(l), 1):
        j = p[i - 1]
        while j > 0 and l[i] != l[j]:
            j = p[j - 1]
        p[i] = j + (l[i] == l[j])
    return s[::-1][:len(s) - p[-1]] + s


# test = 'aacecaaa'
# res = shortestPalindrome(test)
# print res


def findKthLargest(nums, k):
    return nums[k]


test = [3, 2, 1, 5, 6, 4]
res = findKthLargest(test, 2)
print res
