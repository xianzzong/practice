def quick(arry):
    if len(arry)<=1:
        return arry
    length=len(arry)
    mid=arry[int(length/2)]
    left=[]
    midd=[]
    right=[]
    for x in arry:
        if x<mid:
            left.append(x)
        elif x==mid:
            midd.append(x)
        else:
            right.append(x)
    return quick(left)+midd+quick(right)
def twoSum(nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: List[int]
    """
    n=len(nums)
    sort_nums=sorted(nums)
    right=n-1
    left=0
    while(left<right):
        sum=sort_nums[left]+sort_nums[right]
        if (sum==target):
            break
        elif(sum>target):
            right-=1
        else:
            left+=1
    if left==right:
        return 0
    else:
        pos1=nums.index(sort_nums[left])
        pos2=nums.index(sort_nums[right])
        if (pos1==pos2):
            pos2=nums[pos1+1:].index(sort_nums[right])+pos1+1
    return [min(pos1,pos2),max(pos1,pos2)]
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
def addTwoNumbers(l1, l2):
    if l1==None:return l2
    if l2==None:return l1
    flag=0
    mid=ListNode(0)
    p=mid
    while l1 and l2:
        p.next=ListNode((l1.val+l2.val+flag)%10)
        flag=(l1.val+l2.val+flag)/10
        l1=l1.next
        l2=l2.next
        p=p.next
    if l1:
        while l1:
            p.next=ListNode((l1.val+flag)%10)
            flag=(l1.val+flag)/10
            l1=l1.next
            p=p.next
    if l2:
        while l2:
            p.next = ListNode((l2.val + flag) % 10)
            flag = (l2.val + flag) / 10
            l2 = l2.next
            p = p.next
    if flag:
        p.next=ListNode(1)
    return mid.next
def lengthofLongestSubstring(s):
    leng=len(s)
    if leng==0:
        length_max=0
    else:
        length_max=1
    mid=''
    flag=0
    for i in xrange(leng):
        flag=mid.find(s[i])
        if flag==-1:
            mid=mid+s[i]
        else:
            if length_max<len(mid):
                length_max=len(mid)
            mid=mid[flag+1:]+s[i]
    len_mid=len(mid)
    if length_max<len_mid:
        length_max=len_mid
    return length_max
def getKth(numsA,numsB,k):
    lenA=len(numsA)
    lenB=len(numsB)
    if lenA>lenB:
        return getKth(numsB,numsA,k)
    if lenA==0:
        return numsB[k-1]
    if k==1:
        return min(numsA[0],numsB[0])
    pa=min(k/2,lenA)
    pb=k-pa
    if numsA[pa-1]<=numsB[pb-1]:
        return getKth(numsA[pa:],numsB,pb)
    else:
        return getKth(numsA,numsB[pb:],pa)

def findMedianSortedArrays(nums1, nums2):
    len1=len(nums1)
    len2=len(nums2)
    if (len1+len2)%2==1:
        median=getKth(nums1,nums2,(len1+len2)/2+1)
    else:
        median1=getKth(nums1,nums2,(len1+len2)/2)
        median2=getKth(nums1,nums2,(len1+len2)/2+1)
        median=(median1+median2)*0.5
    return median

    mid=nums1+nums2
    mid=sorted(mid)
    leng=len(mid)
    flag=leng%2
    if flag:
        median = float(mid[leng / 2])
    else:
        median=(mid[leng/2-1]+mid[leng/2])/2.0
    return median
def longestPalindrome(s):
    size = len(s)
    if size == 1:
        return s
    newS = ''
    for str in s:
        newS += '#'
        newS += str
    newS += '#'
    maxnum = 0
    maxindex = 0
    mark = [1]
    P = 0  # the rightest index of palindrome
    P0 = 0  # the center while get P
    i = 1
    newSize = len(newS)
    while i < newSize:
        if P == newSize - 1:
            break
        if i < P and mark[2 * P0 - i] < P - i:
            mark.append(mark[2 * P0 - i])
        else:
            if i >= P:
                t = 1
            else:
                t = P - i + 1
            while i - t >= 0 and i + t < newSize:
                if newS[i - t] != newS[i + t]:
                    break
                t += 1
            P0 = i
            P = i + t - 1
            if (maxnum < P - P0 + 1):
                maxnum = P - P0 + 1
                maxindex = P0
            mark.append(P - P0 + 1)
        i += 1
    return s[(maxindex + 1 - maxnum) /2:(maxindex + maxnum) / 2]
def convert(s,numRows):
    len_s=len(s)
    if numRows==1 or len_s<=numRows:
        return s
    new_s=''
    for i in xrange(numRows):
        j=i
        if i==0 or i==numRows-1:
           while j<len_s:
               new_s+=s[j]
               j+=2*(numRows-1)
        else:
            while j<len_s:
                new_s+=s[j]
                j+=2*(numRows-1-i)
                if j>=len_s:
                    break
                else:
                    new_s+=s[j]
                    j+=2*i
    return new_s
def reverse(x):
    flag=1
    if x<0:
        flag=-1
    x=abs(x)
    mid=str(x)
    mid=mid[::-1]
    x_reverse=float(mid)
    if x_reverse>2147483647:
        return 0
    else:
        return flag*int(x_reverse)
def myatoi(str):
    len_str=len(str)
    if len_str==0:
        return 0
    result=0
    flag_0=True
    flag=True
    flag_judge=True
    for i in xrange(len_str):
        if str[i]==' ' and flag_0:
            continue
        if str[i]!=' ':
            flag_0=False
        if flag_judge:
            if str[i]=='+':
                flag_judge=False
                continue
            elif str[i]=='-':
                flag=False
                flag_judge=False
                continue
        if str[i]>='0' and str[i]<='9':
            result=10*result+int(str[i])
            if result>2147483647 and flag:
                return 2147483647
            elif result>2147483648 and not flag:
                return -2147483648
        else:
            break
    if flag:
        return result
    return -1*result
def isPalindrome(x):
    if x<0:
        return False
    b=x
    result=0
    while x>0:
        result=result*10+x%10
        if result>2147483647:
            return False
        x=x/10
    if b==result:
        return True
    else:
        return False
def isMatch(s,p):
    len_s=len(s);len_p=len(p)
    dp=[[False for _ in xrange(len_p+1)] for _ in xrange(len_s+1)]
    dp[0][0]=True
    for i in xrange(1,len_p+1):
        if p[i-1]=='*' and i>=2:
            dp[0][i]=dp[0][i-2]
    for i in xrange(1,len_s+1):
        for j in xrange(1,len_p+1):
            if p[j-1]=='.':
                dp[i][j]=dp[i-1][j-1]
            elif p[j-1]=='*':
                dp[i][j]=dp[i][j-1] or dp[i][j-2] or (dp[i-1][j] and (s[i-1]==p[j-2] or p[j-2]=='.'))
            else:
                dp[i][j]=dp[i-1][j-1] and s[i-1]==p[j-1]
    return dp[len_s][len_p]
def maxArea(height):
    leng=len(height)
    left=0
    right=leng-1
    result=0
    while right>left:
        result=max(result,min(height[left],height[right])*(right-left))
        if height[left]<height[right]:
            k=left
            while height[k]<=height[left] and k <right:
                k+=1
            left=k
        else:
            k=right
            while height[k]<=height[right] and k>left:
                k-=1
            right=k
    return result
def inttoRoman(num):
    val=[1000,900,500,400,100,90,50,40,10,9,5,4,1]
    sym=["M", "CM", "D", "CD", "C", "XC", "L", "XL",  "X", "IX", "V", "IV",  "I"]
    roman=''
    i=0
    while num>0:
        for _ in xrange(num/val[i]):
            roman+=sym[i]
            num-=val[i]
        i+=1
    return roman
def romantointer(s):
    val=[1000,900,500,400,100,90,50,40,10,9,5,4,1]
    sym=["M", "CM", "D", "CD", "C", "XC", "L", "XL",  "X", "IX", "V", "IV",  "I"]
    inte=0
    len_s=len(s)
    maxn=1
    for i in xrange(len_s-1,-1,-1):
        if val[sym.index(s[i])]>=maxn:
            maxn=val[sym.index(s[i])]
            inte+=val[sym.index(s[i])]
        else:
            inte-=val[sym.index(s[i])]
    return inte
def longestCommonPrefix(strs):
    len_strs=len(strs)
    if len_strs==0:
        return ''
    if len_strs==1:
        return strs[0]
    result=strs[0]
    for i in xrange(1,len_strs):
        j=0
        minlen=min(len(result),len(strs[i]))
        while j<minlen:
            if result[j]!=strs[i][j]:
                break
            j+=1
        if j==0:
            return ''
        result=result[:j]
    return result
def threesum(nums):
    len_nums=len(nums)
    res=[]
    if len_nums<3:
        return res
    nums.sort()
    i=0
    while i<len_nums-2:
        j=i+1
        k=len_nums-1
        tmp=0-nums[i]
        while j<k:
            if nums[j]+nums[k]<tmp:
                j+=1
            elif nums[j]+nums[k]>tmp:
                k-=1
            else:
                res.append([nums[i],nums[j],nums[k]])
                j+=1
                k-=1
                while j<k:
                    if nums[j]!=nums[j-1]:
                        break
                    j+=1
        i+=1
        while i<len_nums-2:
            if nums[i]!=nums[i-1]:
                break
            i+=1
    return res
def thresssumclosest(nums,target):
    len_nums=len(nums)
    nums.sort()
    if len_nums<3:
        return 0
    i=0
    res=nums[0]+nums[1]+nums[len_nums-1]
    while i<len_nums-2:
        j=i+1
        k=len_nums-1
        tmp=target-nums[i]
        while j<k:
            if nums[j]+nums[k]==tmp:
                return target
            if nums[j]+nums[k]>tmp:
                if nums[j]+nums[j+1]>=tmp:
                    if nums[j]+nums[j+1]-tmp<abs(res-target):
                        res=nums[i]+nums[j]+nums[j+1]
                    break
                tmpres=nums[i]+nums[j]+nums[k]
                if tmpres-target<abs(res-target):
                    res=tmpres
                k-=1
            else:
                if nums[k]+nums[k-1]<=tmp:
                    if tmp-nums[k]-nums[k-1]<abs(res-target):
                        res=nums[i]+nums[k]+nums[k-1]
                    break
                tmpres=nums[i]+nums[j]+nums[k]
                if target-tmpres<abs(res-target):
                    res=tmpres
                j+=1
        i+=1
        if res==target:
            return target
    return res
# S = [-10,0,-2,3,-8,1,-10,8,-8,6,-7,0,-7,2,2,-5,-8,1,-4,6]
# t=18
# ans=thresssumclosest(S,t)
# print (ans)
def adddigit(digit,res):
    tmp=[]
    for item in digit:
        if len(res)==0:
            tmp.append(item)
        for s in res:
            tmp.append(s+item)
    return tmp
def letterCombinations(digits):
        res=[]
        dic={'0':' ','1':'*','2':'abc','3':'def','4':'ghi','5':'jkl','6':'mno','7':'pqrs','8':'tuv','9':'wxyz'}
        for element in digits:
            tmp=[]
            # res=adddigit(dic[element],res)
            for item in dic[element]:
                if len(res)==0:
                    tmp.append(item)
                for s in res:
                    tmp.append(s+item)
            res=tmp
        return res
# test='23'
# res=letterCombinations(test)
# print res
def foursum(nums,target):
        len_nums=len(nums)
        res=[]
        if len_nums<4:
            return []
        nums.sort()
        i=0
        while i<len_nums-3:
            j=i+1
            while j<len_nums-2:
                mid=target-nums[i]-nums[j]
                k=j+1
                t=len_nums-1
                while k<t:
                    if nums[k]+nums[k+1]>mid:
                        break
                    if nums[t]+nums[t-1]<mid:
                        break
                    if nums[k]+nums[t]==mid:
                        res.append([nums[i],nums[j],nums[k],nums[t]])
                        k+=1
                        t-=1
                        while k<t:
                            if nums[k]==nums[k-1]:
                                k+=1
                            if nums[t]==nums[t+1]:
                                t-=1
                            if nums[k]!=nums[k-1] and nums[t]!=nums[t+1]:
                                break
                    if nums[k]+nums[t]<mid:
                        k+=1
                    if nums[k]+nums[t]>mid:
                        t-=1
                j+=1
                while j<len_nums-2:
                    if nums[j]!=nums[j-1]:
                        break
                    j+=1
            i+=1
            while i<len_nums-3:
                if nums[i]!=nums[i-1]:
                    break
                i+=1
        return res
# test=[5,5,3,5,1,-5,1,-2]
# target=4
# ans=foursum(test,target)
# print ans
def removeNthFromEn(head,n):
        res=ListNode(0)
        res.next=head
        tmp1=res
        tmp2=res
        i=0
        while i<n:
            tmp1=tmp1.next
            i+=1
        while tmp1.next!=None:
            tmp1=tmp1.next
            tmp2=tmp2.next
        tmp2.next=tmp2.next.next
        return res.next
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
def isValid(s):
        len_s=len(s)
        a=len_s%2
        if a!=0:
            return False
        res=[]
        for i in s:
            if i=='(' or i=='[' or i =='{':
                res.append(i)
            if i==')' or i==']' or i=='}':
                if len(res)==0:
                    return False
                tmp=res.pop()
                if i==')' and tmp!='(':
                    return False
                if i==']' and tmp!='[':
                    return False
                if i=='}' and tmp!='{':
                    return False
        if len(res)==0:
            return True
        else:
            return False
test='()[]{}'
res=isValid(test)
print res

