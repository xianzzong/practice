def maxProfit(prices):
    max_profit=0
    len_p=len(prices)
    if len_p==0:
        return max_profit
    lowest_price=prices[0]
    for i in range(1,len_p):
        if prices[i]<lowest_price:
            lowest_price=prices[i]
        else:
            max_profit=max(max_profit,prices[i]-lowest_price)
    return max_profit
# test=[7, 1, 5, 3, 6, 4]
# res=maxProfit(test)
# print res
def maxProfit2(prices):
    max_profit=0
    if len(prices)==0:
        return max_profit
    for i in range(1,len(prices)):
        if prices[i]>=prices[i-1]:
            max_profit+=prices[i]-prices[i-1]
    return max_profit
# test=[1,2,3,4,5]
# res=maxProfit2(test)
# print res
def maxProfit3(prices):
    len_p=len(prices)
    if len_p==0:
        return 0
    tmp1=[0 for i in range(len_p)]
    tmp2=tmp1[:]
    tmp1[0]=0
    lowest=prices[0]
    for i in range(1,len_p):
        lowest=min(lowest,prices[i])
        tmp1[i]=max(tmp1[i-1],prices[i]-lowest)
    tmp2[len_p-1]=0;lowest=prices[len_p-1]
    for i in range(len_p-2,-1,-1):
        lowest=max(lowest,prices[i])
        tmp2[i]=max(tmp2[i+1],lowest-prices[i])
    res=0
    for i in range(len_p):
        res=max(res,tmp1[i]+tmp2[i])
    return res
# test=[7, 1, 5, 3, 6, 4]
# res=maxProfit3(test)
# print res
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution(object):
    def maxPathSum(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """

        def solve(root):
            if root == None:
                return 0
            sum_s = 0
            solve_l = 0;
            solve_r = 0
            if root.left:
                solve_l = solve(root.left)
                if solve_l > 0:
                    sum_s += solve_l
            if root.right:
                solve_r = solve(root.right)
                if solve_r > 0:
                    sum_s += solve_r
            Solution.max = max(Solution.max, sum_s)
            return max(root.val, max(root.val + solve_l, root.val + solve_r))

        if root == None:
            return 0
        else:
            Solution.max = root.val
            solve(root)
            return Solution.max
def isPalindrome(s):
    len_s=len(s)
    if len_s==0:
        return True
    left=0;right=len_s-1
    while left<right:
        print s[left],s[right]
        if s[left].isalnum() and s[right].isalnum():
            if s[left].lower()==s[right].lower():
                left+=1
                right-=1
            else:
                return False
        elif s[left].isalnum():
            right-=1
        elif s[right].isalnum():
            left+=1
        else:
            left+=1
            right-=1
    return True
# test=",."
# res=isPalindrome(test)
# print (res)
def ladderLength(beginWord, endWord, wordList):
    if endWord not in wordList:
        return 0
    wordDict=set(wordList)
    start=set([beginWord])
    end=set([endWord])
    length=1
    while start:
        if start & end:
            return length
        start=wordDict & (set(word[:i]+cha+word[i+1:]
                              for word in start
                              for i in range(len(beginWord)) for cha in 'abcdefghijklmnopqrstuvwxyz'))
        length+=1
        if len(start)>len(end):
            start,end=end,start
        wordDict-=start
    return 0
#    wordList.append(endWord)
#     tmp_list=[]
#     tmp_list.append([beginWord,1])
#     while tmp_list:
#         tmp=tmp_list.pop(0)
#         current_word=tmp[0]
#         current_idex=tmp[1]
#         if current_word==endWord:
#             return current_idex
#         for i in range(len(beginWord)):
#             for j in 'abcdefghijklmnopqrstuvwxyz':
#                 if current_word[i]!=j:
#                     newword=current_word[:i]+j+current_word[i+1:]
#                     if newword in wordList:
#                         tmp_list.append([newword,current_idex+1])
#                         wordList.remove(newword)
#     return 0
#     wordDict = set(wordList)
#     if endWord not in wordList:
#         return 0
#     length = 1
#     front, back = set([beginWord]), set([endWord])
#     wordDict.discard(beginWord)
#     while front:
#         if front & back:
#             # there are common elements in front and back, done
#             return length
#         length += 1
#         # generate all valid transformations
#         front = wordDict & (set(word[:index] + ch + word[index + 1:] for word in front
#                                 for index in range(len(beginWord)) for ch in 'abcdefghijklmnopqrstuvwxyz'))
#         if len(front) > len(back):
#             # swap front and back for better performance (fewer choices in generating nextSet)
#             front, back = back, front
#         # remove transformations from wordDict to avoid cycle
#         wordDict -= front
#     return 0
def findLadders(beginWord, endWord, wordList):
    def getpath(path,word):
        if word==beginWord:
            path.append(word)
            tmp=path[:]
            tmp.reverse()
            result.append(tmp)
            path.pop()
            return
        path.append(word)
        for item in transition[word]:
            getpath(path,item)
        path.pop()
    wordDict=set(wordList)
    transition={}
    for item in wordDict:
        transition[item]=[]
    tmp1=set();tmp2=set()
    length=len(beginWord)
    tmp1.add(beginWord)
    while True:
        for item in tmp1:
            for i in range(length):
                for j in 'abcdefghijklmnopqrstuvwxyz':
                    if j!=item[i]:
                        newword=item[:i]+j+item[i+1:]
                        if newword in wordDict:
                            tmp2.add(newword)
                            transition[newword].append(item)
        wordDict-=tmp2
        if endWord in tmp2:
            break
        if len(tmp2)==0:
            return []
        tmp1=tmp2.copy()
        tmp2.clear()
    result=[]
    getpath([],endWord)
    return result
    # def buildpath(path, word):
    #     if len(prevMap[word]) == 0:
    #         path.append(word);
    #         currPath = path[:]
    #         currPath.reverse();
    #         result.append(currPath)
    #         path.pop();
    #         return
    #     path.append(word)
    #     for iter in prevMap[word]:
    #         buildpath(path, iter)
    #     path.pop()
    #
    # result = []
    # prevMap = {}
    # length = len(beginWord)
    # wordList.append(beginWord)
    # wordDict=set(wordList)
    # for i in wordDict:
    #     prevMap[i] = []
    # candidates = [set(), set()]
    # current = 0
    # previous = 1
    # candidates[current].add(beginWord)
    # while True:
    #     current, previous = previous, current
    #     for i in candidates[previous]: wordDict.remove(i)
    #     candidates[current].clear()
    #     for word in candidates[previous]:
    #         for i in range(length):
    #             part1 = word[:i];
    #             part2 = word[i + 1:]
    #             for j in 'abcdefghijklmnopqrstuvwxyz':
    #                 if word[i] != j:
    #                     nextword = part1 + j + part2
    #                     if nextword in wordDict:
    #                         prevMap[nextword].append(word)
    #                         candidates[current].add(nextword)
    #     if len(candidates[current]) == 0: return result
    #     if endWord in candidates[current]: break
    # buildpath([], endWord)
    # return result

# beginWord="hot"
# endWord="dog"
# wordList=["hot","dog"]
# res=findLadders(beginWord,endWord,wordList)
# print res
def longestConsecutive(nums):
    maxlen=0
    len_n=len(nums)
    if len_n==0:
        return maxlen
    result={item:False for item in nums}
    for item in nums:
        if result[item]==True:
            continue
        else:
            result[item]=True
            item_left=0;item_right=0
            tmp=item-1
            while tmp in nums:
                result[tmp]=True
                item_left+=1
                tmp-=1
            tmp=item+1
            while  tmp in nums:
                result[tmp]=True
                item_right+=1
                tmp+=1
            maxlen=max(maxlen,item_right+1+item_left)
    return maxlen
# test=[100, 4, 200, 1, 3, 2]
# res=longestConsecutive(test)
# print (res)
def sortedArrayToBST(nums):
    """
    :type nums: List[int]
    :rtype: TreeNode
    """
    if len(nums)==0:
        return None
    res=TreeNode(0)
    mid_index=len(nums)/2
    mid=nums[mid_index]
    res.val=mid
    res.left=sortedArrayToBST(nums[:mid_index])
    res.right=sortedArrayToBST(nums[mid_index+1:])
    return res

def sumNumbers(root):
    result = []
    def getnum(root):
        if root.left == None and root.right == None:
            result.append(root.val)
        elif root.left==None:
            root.right.val+=10*root.val
            getnum(root.right)
        elif root.right==None:
            root.left.val+=10*root.val
            getnum(root.left)
        else:
            root.right.val+=10*root.val
            getnum(root.right)
            root.left.val+=10*root.val
            getnum(root.left)
    if root==None:
        return 0
    getnum(root)
    return sum(result)
# test=sortedArrayToBST([1,2,3,5,6,7,8,9])
# res=sumNumbers(test)
# print res
def solve(board):
    def find_O(x,y,board):
        board[x]=board[x][:y]+'D'+board[x][y+1:]
        if 0<x-1<row and board[x-1][y]=='O':
            find_O(x-1,y,board)
        if 0<x+1<row and board[x+1][y]=='O':
            find_O(x+1,y,board)
        if 0<y-1<col and board[x][y-1]=='O':
            find_O(x,y-1,board)
        if 0 < y +1 < col and board[x][y+1] == 'O':
            find_O(x, y+1, board)
    row=len(board)
    if row==0:
        return
    col=len(board[0])
    for i in range(col):
        if board[0][i]=='O':
            find_O(0,i,board)
        if board[row-1][i]=='O':
            find_O(row-1,i,board)
        i+=1
    for j in range(1,row-1):
        if board[j][0]=='O':
            find_O(j,0,board)
        if board[j][col-1]=='O':
            find_O(j,col-1,board)
    for i in range(row):
        for j in range(col):
            if board[i][j]=='O':
                board[i] = board[i][:j] + 'X' + board[i][j + 1:]
            if board[i][j]=='D':
                board[i] = board[i][:j] + 'O' + board[i][j + 1:]
# test=["XXXX","XOOX","XXOX","XOXX"]
# solve(test)
# print test
def partition(s):
    def ispalindrome(subs):
        for i in range(len(subs)):
            if subs[i]!=subs[len(subs)-1-i]:return False
        return True
    def solve(tmps,tmpresulet):
        if len(tmps)==0:
            result.append(tmpresulet)
        for i in range(1,len(tmps)+1):
            # tmpps=tmps[:i]
            # tmppps=tmps[i:]
            if ispalindrome(tmps[:i]):
                solve(tmps[i:],tmpresulet+[tmps[:i]])
    result=[]
    solve(s,[])
    return result
# res=partition('aab')
# print res
def minCut(s):
    len_s=len(s)
    result=[i for i in range(len_s)]
    tmp=[[False for i in range(len_s)] for i in range(len_s)]
    tmp[0][0]=True
    j=1
    while j<len_s:
        tmp[j][j]=True
        i=j-1
        result[j]=min(result[j],result[j-1]+1)
        while i>=0:
            if s[i]==s[j] and((j-i)<2 or tmp[i+1][j-1]):
                tmp[i][j]=True
                if i==0:
                    result[j]=0
                else:
                    result[j]=min(result[j],result[i-1]+1)
            i-=1
        j+=1
    return result[len_s-1]
# res=minCut('efe')
# print res
class UndirectedGraphNode:
    def __init__(self, x):
        self.label = x
        self.neighbors = []

# @param node, a undirected graph node
# @return a undirected graph node
def cloneGraph(self, node):
    if node==None:
        return node
    d={}
    def solve(n):
        if n in d:
            return d[n]
        tmp=UndirectedGraphNode(n.label)
        d[n]=tmp
        for item in n.neighbors:
            tmp.neighbors.append(solve(item))
        return tmp
    return solve(node)

def canCompleteCircuit(gas, cost):
    sum_g=sum(gas)
    sum_c=sum(cost)
    if sum_g<sum_c:
        return -1
    tmpsum=0;begin=0
    for i in range(len(gas)):
        tmpsum=gas[i]-cost[i]+tmpsum
        if tmpsum<0:
            tmpsum=0
            begin=i+1
        i+=1
    return begin
# test=[[1,2],[2,1]]
# res=canCompleteCircuit(test[0],test[1])
# print res
def candy(ratings):
    len_r=len(ratings)
    candy_num=[1 for i in range(len_r)]
    for i in range(1,len_r):
        if ratings[i]>ratings[i-1]:
            candy_num[i]=candy_num[i-1]+1
    for i in range(len_r-1,0,-1):
        if ratings[i]<ratings[i-1] and candy_num[i]>=candy_num[i-1]:
            candy_num[i-1]=candy_num[i]+1
    return sum(candy_num)
# test=[4,5,6,2,3,8,9,12,4,2,5,1,23,4,11,8,4,3,8,12]
# res=candy(test)
# print res
def singleNumber(nums):
    res=0
    for item in nums:
        res=res^item
    return res
# test=[1,1,2,2,3,3,4,5,5,6,6,7,7,8,8]
# res=singleNumber(test)
# print res
def singleNumber2(nums):
    one,two,three=0,0,0
    for item in nums:
        two=two|(one & item)
        one=one^item
        three=~(one & two)
        one=three&one
        two=three&two
    return one
# test=[3, 3, 3, 2, 2, 2,1]
# res=singleNumber2(test)
# print res
class RandomListNode(object):
    def __init__(self, x):
        self.label = x
        self.next = None
        self.random = None

class Solution(object):
    def copyRandomList(self, head):
        """
        :type head: RandomListNode
        :rtype: RandomListNode
        """
        if head==None:
            return head
        tmp=head
        while tmp:
            newnode=RandomListNode(tmp.label)
            newnode.next=tmp.next
            tmp.next=newnode
            tmp=tmp.next.next
        tmp=head
        while tmp:
            if tmp.random:
                tmp.next.random=tmp.random.next
            tmp=tmp.next.next
        NewNode=head.next
        pold=head
        pnew=NewNode
        while pnew.next:
            pold.next=pnew.next
            pold=pold.next
            pnew.next=pold.next
            pnew=pnew.next
        pold.next=None
        pnew.next=None
        return NewNode
def wordBreak(s, wordDict):
    len_s=len(s)
    result=[False for i in range(len_s+1)]
    result[0]=True
    for i in range(1,len_s+1):
        for j in range(i):
            tmp=s[j:i]
            if result[j] and s[j:i] in wordDict:
                result[i]=True
                break
    return result[len_s]
# s = "leetcode"
# dicts = ["leet", "code"]
# res=wordBreak(s,dicts)
# print res
def wordBreak2(s, wordDict):
    def isbreak(s,wordDict):
        len_s = len(s)
        result = [False for i in range(len_s + 1)]
        result[0] = True
        for i in range(1, len_s + 1):
            for j in range(i):
                tmp = s[j:i]
                if result[j] and s[j:i] in wordDict:
                    result[i] = True
                    break
        return result[len_s]
    res=[]
    tmp=''
    if not isbreak(s,wordDict):
        return []
    if s in wordDict:
        res.append(s)
    for i in range(1,len(s)+1):
        tmp=s[:i]
        if tmp in wordDict:
            if isbreak(s[i:],wordDict):
                tmp_s=wordBreak2(s[i:],wordDict)
                for item in tmp_s:
                    res.append(tmp+' '+item)
    return res
s = "aaaaaaa"
dicts = ["aaaa", "aa", "a"]
res=wordBreak2(s,dicts)
print res