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


if __name__ == "__main__":
    solu = Solution()
    s = "anagram"
    t = "nagaram"
    res = solu.isAnagram(s, t)
    print (res)
