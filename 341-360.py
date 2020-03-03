# -*- coding: utf-8 -*
# 341. Flatten Nested List Iterator
# """
# This is the interface that allows for creating nested lists.
# You should not implement it, or speculate about its implementation
# """


from sortedcontainers import SortedList


class NestedInteger:
    def isInteger(self) -> bool:
        """
        @return True if this NestedInteger holds a single integer, rather than a nested list.
        """

    def getInteger(self):
        """
        @return the single integer that this NestedInteger holds, if it holds a single integer
        Return None if this NestedInteger holds a nested list
        """

    def getList(self):
        """
        @return the nested list that this NestedInteger holds, if it holds a nested list
        Return None if this NestedInteger holds a single integer
        """
        return []


class NestedIterator:
    def __init__(self, nestedList: [NestedInteger]):
        self.stack = nestedList[::-1]

    def next(self) -> int:
        return self.stack.pop().getInteger()

    def hasNext(self) -> bool:
        while self.stack and not self.stack[-1].isInteger():
            self.stack.extend(self.stack.pop().getList()[::-1])
        return len(self.stack) != 0

        # Your NestedIterator object will be instantiated and called as such:
        # i, v = NestedIterator(nestedList), []
        # while i.hasNext(): v.append(i.next())
# 342. Power of Four


class Solution:
    def isPowerOfFour(self, num: int) -> bool:
        if num == 1:
            return True
        if num < 4:
            return False
        while num > 1:
            tmp = num % 4
            if tmp != 0:
                return False
            else:
                num = int(num / 4)
        return True


# res = solu.isPowerOfFour(8)
# 343. Integer Break
class Solution:
    def integerBreak(self, n: int) -> int:

        # 可以说，拆成3的比拆成2的乘积大。 比如6的时候 2*2*2 < 3*3

        # 我们希望能尽可能的拆成3，然后才是2.

        # 所以，如果

        # n % 3 == 0:  那么全部拆成3
        # n % 3 == 1:  2个2剩下的为3    4*3 ^ (x-1) > 1*3 ^ x
        # n % 3 == 2:  1个2剩下的为3
        if n <= 3:
            return n-1
        mod = n % 3
        if mod == 0:
            return int(pow(3, n / 3))
        elif mod == 1:
            return int(4 * pow(3, (n - 4) / 3))
        else:
            return int(2 * pow(3, (n - 2) / 3))


# 344. Reverse String
class Solution:
    def reverseString(self, s):
        """
        Do not return anything, modify s in-place instead.
        """
        return s.reverse()


# 345. Reverse Vowels of a String
class Solution:
    def reverseVowels(self, s: str) -> str:
        left = 0
        right = len(s) - 1
        check = "aeiouAEIOU"
        while left < right:
            while left < right and s[left] not in check:
                left += 1
            while right > left and s[right] not in check:
                right -= 1

            if left >= right:
                break

            tmp1 = s[left]
            tmp2 = s[right]

            s = s[:left] + tmp2 + s[left + 1:right] + tmp1 + s[right + 1:]
            left += 1
            right -= 1
        return s


# 347. Top K Frequent Elements
class Solution:
    def topKFrequent(self, nums, k):
        import collections
        import heapq
        count = collections.Counter(nums)
        heap = []
        for key, cnt in count.items():
            if len(heap) < k:
                heapq.heappush(heap, (cnt, key))
            else:
                if heap[0][0] < cnt:
                    heapq.heappop(heap)
                    heapq.heappush(heap, (cnt, key))
        res = [x[1] for x in heap]
        return res


# res = solu.topKFrequent([1, 1, 1, 2, 2, 3], 2)

# 349. Intersection of Two Arrays
class Solution:
    def intersection(self, nums1, nums2):
        return set(nums1) & set(nums2)


# 350. Intersection of Two Arrays II
class Solution:
    def intersect(self, nums1, nums2):
        import collections
        res = []
        nums1 = collections.Counter(nums1)
        for item in nums2:
            if item in nums1:
                res.append(item)
                nums1[item] -= 1
                if nums1[item] == 0:
                    del nums1[item]
        return res

# 352. Data Stream as Disjoint Intervals


class SummaryRanges:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.start_interval = {}  # it's easier to find and pop intervals that are no longer valid
        self.end_interval = {}
        self.processed = set()

    def addNum(self, val: int) -> None:
        if val in self.processed:
            return
        self.processed.add(val)
        start, end = val, val
        if self.start_interval and self.end_interval:
            if val+1 in self.start_interval:
                end = self.start_interval[val+1][1]
                self.start_interval.pop(val+1)
            if val-1 in self.end_interval:
                start = self.end_interval[val-1][0]
                self.end_interval.pop(val-1)
        interval = [start, end]
        self.start_interval[start] = interval
        self.end_interval[end] = interval

        def getIntervals(self) -> List[List[int]]:
            return sorted(self.start_interval.values())


# 354. Russian Doll Envelopes
class Solution:
    def maxEnvelopes(self, envelopes):
        def lower_bound(arrays, L, R, x):
            while L < R:
                mid = (L + R) >> 1
                if x <= arrays[mid]:
                    R = mid
                else:
                    L = mid + 1
            return L

        if not envelopes:
            return 0
        envelopes.sort(key=lambda x: (x[0], -x[1]))
        length = len(envelopes)
        dp = [1] * length
        gt = [0x7fffffff] * (length + 1)
        for i in range(length):
            k = lower_bound(gt, 1, length, envelopes[i][1])
            dp[i] = k
            gt[k] = envelopes[i][1]
        return max(dp)


# 355. Design Twitter
class Twitter(object):
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.tweets_cnt = 0
        self.tweets = collections.defaultdict(list)
        self.follower_ship = collections.defaultdict(set)

    def postTweet(self, userId, tweetId):
        """
        Compose a new tweet.
        :type userId: int
        :type tweetId: int
        :rtype: void
        """
        self.tweets[userId].append([tweetId, self.tweets_cnt])
        self.tweets_cnt += 1

    def getNewsFeed(self, userId):
        """
        Retrieve the 10 most recent tweet ids in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user herself. Tweets must be ordered from most recent to least recent.
        :type userId: int
        :rtype: List[int]
        """
        recent_tweets = []
        user_list = list(self.follower_ship[userId]) + [userId]
        userId_tweet_index = [[userId, len(
            self.tweets[userId]) - 1] for userId in user_list if userId in self.tweets]

        for _ in xrange(10):
            max_index = max_tweet_id = max_user_id = -1
            for i, (user_id, tweet_index) in enumerate(userId_tweet_index):
                if tweet_index >= 0:
                    tweet_info = self.tweets[user_id][tweet_index]
                    if tweet_info[1] > max_tweet_id:
                        max_index, max_tweet_id, max_user_id = i, tweet_info[1], user_id

            if max_index < 0:
                break
            recent_tweets.append(
                self.tweets[max_user_id][userId_tweet_index[max_index][1]][0])
            userId_tweet_index[max_index][1] -= 1

        return recent_tweets

    def follow(self, followerId, followeeId):
        """
        Follower follows a followee. If the operation is invalid, it should be a no-op.
        :type followerId: int
        :type followeeId: int
        :rtype: void
        """
        if followerId != followeeId:
            self.follower_ship[followerId].add(followeeId)

    def unfollow(self, followerId, followeeId):
        """
        Follower unfollows a followee. If the operation is invalid, it should be a no-op.
        :type followerId: int
        :type followeeId: int
        :rtype: void
        """
        if followerId in self.follower_ship and followeeId in self.follower_ship[followerId]:
            self.follower_ship[followerId].remove(followeeId)


# 357. Count Numbers with Unique Digits
class Solution:
    def countNumbersWithUniqueDigits(self, n: int) -> int:
        n = min(n, 10)
        result = [1] + [9] * n
        for i in range(2, n + 1):
            for x in range(9, 9 - i + 1, -1):
                result[i] *= x
        return sum(result)


if __name__ == "__main__":
    solu = Solution()
    nums1 = [1, 2, 2, 1]
    nums2 = [2, 2]
    res = solu.maxEnvelopes([[5, 4], [6, 4], [6, 7], [2, 3]])
    print(res)
