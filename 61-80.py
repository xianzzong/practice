class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
def rotateRight(head, k):
        if k==0:
            return head
        if head==None:
            return head
        tmp=ListNode(0)
        tmp.next=head
        p=tmp
        count=0
        while p.next:
            p=p.next
            count+=1
        p.next=tmp.next
        step=count-(k%count)
        for i in range(0,step):
            p=p.next
        head=p.next
        p.next=None
        return head
a,b,c,d,e=ListNode(1),ListNode(2),ListNode(3),ListNode(4),ListNode(5)
a.next=b
b.next=c
c.next=d
d.next=e
res=rotateRight(a,2)
print res
def factorial(n):
        res=1
        if n==0:
            return 0
        for i in range(1,n+1):
            res=res*i
        return res
def uniquePaths(m, n):
        if m==1 or n==1:
            return 1
        sum_n_1=factorial(n-1)
        sum_m_1=factorial(m-1)
        sum_m_n_2=factorial(n+m-2)
        res=sum_m_n_2/(sum_m_1*sum_n_1)
        return res
# res=uniquePaths(3,3)
# print res
def uniquePathsWithObstacles(obstacleGrid):
        row=len(obstacleGrid)
        col=len(obstacleGrid[0])
        res=[[0 for i in range(col)] for i in range(row)]
        for i in range(row):
            if obstacleGrid[i][0]==0:
                res[i][0]=1
            else:
                break
        for i in range(col):
            if obstacleGrid[0][i]==0:
                res[0][i]=1
            else:
                break
        for i in range(1,row):
            for j in range(1,col):
                if obstacleGrid[i][j]==1:
                    res[i][j]=0
                else:
                    res[i][j]=res[i-1][j]+res[i][j-1]
        return res[row-1][col-1]
# test=[
#   [0,0,0],
#   [0,1,0],
#   [0,0,0]
# ]
# res=uniquePathsWithObstacles(test)
# print (res)
def minPathSum(grid):
    row=len(grid)
    col=len(grid[0])
    res=[[0 for i in range(col)] for i in range(row)]
    res[0][0]=grid[0][0]
    for i in range(1,row):
        res[i][0]=res[i-1][0]+grid[i][0]
    for j in range(1,col):
        res[0][j]=res[0][j-1]+grid[0][j]
    for i in range(1,row):
        for j in range(1,col):
            res[i][j]=min(res[i-1][j],res[i][j-1])+grid[i][j]
    return res[row-1][col-1]
# test=[
#   [1,2,3],
#   [3,1,4],
#   [3,3,2]
# ]
# res=minPathSum(test)
# print (res)
def isNumber(s):
        begin,last=0,len(s)-1
        while begin<=last and s[begin]==' ':
            begin+=1
        while begin<=last and s[last]==' ':
            last-=1
        if begin<=last and (s[begin]=='+' or s[begin]=='-'):
            begin+=1
        flag_number,flag_exp,flag_dot=False,False,False
        while begin<=last:
            if s[begin]>='0' and s[begin]<='9':
                flag_number=True
            elif s[begin]=='.':
                if flag_exp or flag_dot:
                    return False
                else:
                    flag_dot=True
            elif s[begin]=='e' or s[begin]=='E':
                if flag_exp or not flag_number:
                    return False
                else:
                    flag_exp=True;flag_number=False
            elif s[begin]=='+' or s[begin]=='-':
                if s[begin-1]!='e' and s[begin-1]!='E':
                    return False
            else:
                return False
            begin+=1
        return flag_number
# test='005047e+6'
# res=isNumber(test)
# print res
def plusOne(digits):
        flag=0
        len_d=len(digits)
        digits[len_d-1]+=1
        for i in range(len_d-1,-1,-1):
            digits[i]=digits[i]+flag
            flag=0
            if digits[i]>9:
                flag=1
                digits[i]=digits[i]%10
        if flag:
            digits.insert(0,1)
        return digits
# test=[8,9,9,9]
# res=plusOne(test)
# print res
def addBinary(a, b):
        len_a=len(a);len_b=len(b)
        i_a=len_a-1;i_b=len_b-1
        res=''
        if len_a==0:
            return b
        if len_b==0:
            return a
        flag=0
        while i_a>=0 and i_b>=0:
            tmp=int(a[i_a])+int(b[i_b])+flag
            flag=0
            if tmp>1:
                flag=1
                tmp=tmp%2
            res=str(tmp)+res
            i_a-=1;i_b-=1
        while i_a>=0:
            tmp=int(a[i_a])+flag
            flag=0
            if tmp>1:
                flag=1
                tmp=tmp%2
            res=str(tmp)+res
            i_a-=1
        while i_b>=0:
            tmp=int(b[i_b])+flag
            flag=0
            if tmp>1:
                flag=1
                tmp=tmp%2
            res=str(tmp)+res
            i_b-=1
        if flag:
            res=str(flag)+res
        return res
# test=['11','111']
# res=addBinary(test[0],test[1])
# print res
def fullJustify(words, maxWidth):
        number_words=len(words)
        first=0;last=0     #  every row have words[first]-words[last-1]
        res=[]
        tmp=''             #  tmp wordlist
        i=0
        while i<number_words:
            if len(tmp)==0:           # judge if the length >maxWidth
                judge=tmp+words[i]
            else:
                judge=tmp+' '+words[i]
            if len(judge)>maxWidth:
                word_num=last-first-1 #number -1
                if word_num!=0: # just one
                    space_total=maxWidth-len(tmp)+word_num # total space
                    space_num=space_total/(word_num)       # the number of position to insert space
                    space_num_1=space_total%(word_num)     # the rest of space
                    tmp_1=words[first]
                    for j in range(word_num):
                        if space_num_1>0:
                            tmp_1=tmp_1+' '*(space_num+1)+words[first+j+1]
                            space_num_1-=1
                        else:
                            tmp_1=tmp_1+' '*space_num+words[first+j+1]
                    res.append(tmp_1)
                else:
                    space_total = maxWidth - len(tmp)
                    tmp=tmp+' '*space_total
                    res.append(tmp)
                tmp = ''
                first = i;last = i
            else:
                last+=1
                if tmp=='':
                    tmp=tmp+words[i]
                else:
                    tmp=tmp+' '+words[i]
                i+=1
        res.append(tmp)
        len_r = len(res)
        space_num = maxWidth - len(res[len_r - 1])
        res[len_r - 1] += ' ' * space_num
        return res
# words=["Don't","go","around","saying","the","world","owes",
#        "you","a","living;","the","world",
#         "owes","you","nothing;","it","was","here","first."]
# L=30
# res=fullJustify(words,L)
# print res
def mySqrt(x):
        if x==0:
            return 0
        first=1;last=x
        while abs(last-first)>1:
            mid=(first+last)/2
            tmp=mid**2
            if tmp==x:
                return mid
            elif tmp>x:
                last=mid
            else:
                first=mid
        return first
# res=mySqrt(25)
# print res
def climbStairs(n):
        res=[1,2]
        for i in range(2,n):
            tmp=res[i-1]+res[i-2]
            res.append(tmp)
        return res[n-1]
# res=climbStairs(4)
# print res
def simplifyPath(path):
        res=[];len_p=len(path)
        i=0
        while i<len_p:
            end=i+1
            while end<len_p and path[end]!='/':
                end+=1
            tmp_str=path[i:end]
            if tmp_str=='/..':
                if len(res)>0:
                    res.pop()
            elif tmp_str!='/.' and tmp_str!='/':
                res.append(tmp_str)
            i=end
        if len(res)==0:
            return '/'
        return ''.join(res)
# test="///"
# res=simplifyPath(test)
# print res
def minDistance(word1, word2):
    len_word1=len(word1)+1;len_word2=len(word2)+1
    res=[[0 for i in range(len_word2)] for i in range(len_word1)]
    for i in range(len_word1):
        res[i][0]=i
    for j in range(len_word2):
        res[0][j]=j
    for i in range(1,len_word1):
        for j in range(1,len_word2):
            if word1[i-1]==word2[j-1]:
                res[i][j]=res[i-1][j-1]
            else:
                res[i][j]=min(res[i-1][j],res[i][j-1],res[i-1][j-1])+1
    return res[len_word1-1][len_word2-1]
# test=['b','a']
# res=minDistance(test[0],test[1])
# print res
def setZeroes(matrix):
        row=len(matrix)
        col=len(matrix[0])
        row_flag=[False for i in range(row)]
        col_flag=[False for i in range(col)]
        for i in range(row):
            for j in range(col):
                if matrix[i][j]==0:
                    row_flag[i]=True
                    col_flag[j]=True

        for i in range(row):
            for j in range(col):
                if row_flag[i] or col_flag[j]:
                    matrix[i][j]=0
# test=[
#   [1,2,3],
#   [0,0,4],
#   [3,3,2]
# ]
# setZeroes(test)
# print test
def searchMatrix(matrix, target):
    row=len(matrix)
    if row==0:
        return False
    col=len(matrix[0])
    if col==0:
        return False
    first=0;last=row-1
    while first<last:
        mid=(first+last+1)/2
        if matrix[mid][0]==target:
            return True
        elif matrix[mid][0]<target:
            first=mid
        else:
            last=mid-1
    target_row=first
    first=0;last=col-1
    while first<=last:
        mid=(first+last)/2
        if matrix[target_row][mid]==target:
            return True
        elif matrix[target_row][mid]<target:
            first=mid+1
        else:
            last=mid-1
    return False
# test=[
# ]
# res=searchMatrix(test,20)
# print res
def sortColors(nums):
    len_n=len(nums)
    last=len_n-1;first=0
    i=0
    while i<=last:
        if nums[i]==0:
            nums[first],nums[i]=nums[i],nums[first]
            first+=1
            i+=1
        elif nums[i]==2:
            nums[last],nums[i]=nums[i],nums[last]
            last-=1
        else:
            i+=1
# test=[1,0,2,1,0,0,1,2,1,2,0,0,0,1,0]
# sortColors(test)
# print test
def minWindow(s, t):
        len_s=len(s);len_t=len(t)
        dict_t={}
        for item in t:
            if item in dict_t:
                dict_t[item]+=1
            else:
                dict_t[item]=1
        minlength_t=len_s+1
        start=0;end=0;res=''
        for end in range(len_s):
            if s[end] in dict_t:
                dict_t[s[end]]-=1
                if dict_t[s[end]]>=0:
                    len_t-=1
            if len_t==0:
                while True:
                    if s[start] in dict_t:
                        if dict_t[s[start]]<0:
                            dict_t[s[start]]+=1
                            start+=1
                        else: break
                    else:
                        start+=1
                sub_str=s[start:end+1]
                if len(sub_str)<minlength_t:
                    minlength_t=len(sub_str)
                    res=sub_str
        return res
# test="AAOBECODEBANC"
# target="ABC"
# res=minWindow(test,target)
# print (res)
def combine(n, k):
    def solve_combine(start,n,k):
        res=[]
        if k==n-start+1:
            t=[]
            for i in range(k):
                t.append(start)
                start+=1
            res.append(t)
            return res
        if k==1:
            while start<=n:
                res.append([start])
                start+=1
            return res
        tmp1=solve_combine(start+1,n,k-1)
        tmp2=solve_combine(start+1,n,k)
        for item in tmp1:
            tmp=item+[start]
            res.append(tmp)
        for item in tmp2:
            res.append(item)
        return res
    if k>n or k<=0:
        return [[]]
    res=solve_combine(1,n,k)
    return res
# res=combine(4,0)
# print res
def subsets(nums):
        res=[]
        len_n=len(nums)
        if len_n==0:
            return [[]]
        tmp=subsets(nums[:len_n-1])
        for item in tmp:
            res.append(item)
            res.append(item+[nums[len_n-1]])
        return res
# test=[1,2,3]
# res=subsets(test)
# print res
def exist(board, word):
    len_word=len(word)
    if len_word==0:
        return True
    row=len(board)
    if row==0:
        return False
    col=len(board[0])
    if col==0:
        return False
    def search_exit(x,y,word_str):
        if len(word_str)==0:
            return True
        # go up
        if x>0 and board[x-1][y]==word_str[0]:
            tmp=board[x][y];board[x][y]='#'
            if search_exit(x-1,y,word_str[1:]):
                return True
            board[x][y]=tmp
        # go down
        if x<row-1 and board[x+1][y]==word_str[0]:
            tmp=board[x][y];board[x][y]='#'
            if search_exit(x+1,y,word_str[1:]):
                return True
            board[x][y]=tmp
        #go left
        if y>0 and board[x][y-1]==word_str[0]:
            tmp=board[x][y];board[x][y]='#'
            if search_exit(x,y-1,word_str[1:]):
                return True
            board[x][y]=tmp
        #go right
        if y<col-1 and board[x][y+1]==word_str[0]:
            tmp=board[x][y];board[x][y]='#'
            if search_exit(x,y+1,word_str[1:]):
                return True
            board[x][y]=tmp
        return False
    for i in range(row):
        for j in range(col):
            if board[i][j]==word[0]:
                if search_exit(i,j,word[1:]):
                    return True
    return False
# test=[
#   ['A','B','C','E'],
#   ['S','F','C','S'],
#   ['A','D','E','E']
# ]
# word='ABCCED'
# res=exist(test,word)
# print res
def removeDuplicates(nums):
        len_n=len(nums)
        if len_n==0:
            return len_n
        flag=1;i=1
        while i<len(nums):
            if nums[i]==nums[i-1]:
                flag+=1
                if flag>2:
                    nums.remove(nums[i])
                    flag-=1
                else:
                    i+=1
            else :
                flag=1
                i+=1
        return len(nums)
test=[1,1,1,2,2,2,2,2,3,3,3,3,4]
res=removeDuplicates(test)
print res