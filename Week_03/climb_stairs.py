# coding:utf-8

# 法一： 1个函数
def climbStairs(n):
    if n <= 2:
        return n
    return climbStairs(n - 1) + climbStairs(n - 2)


# 法二：2个函数
def climbStairs2(n):
    return _climb_stairs2(0, n)


def _climb_stairs2(cur_level, max_level):
    if cur_level > max_level: return 0
    if cur_level == max_level: return 1
    return _climb_stairs2(cur_level + 1, max_level) + _climb_stairs2(cur_level + 2, max_level)


# 法三: +Cache
def climbStairs3(n):
    memo = {}
    return _climb_stairs3(0, n, memo)


def _climb_stairs3(cur_level, max_level, memo):
    if cur_level > max_level: return 0
    if cur_level == max_level: return 1
    if cur_level in memo: return memo[cur_level]

    memo[cur_level] = _climb_stairs3(cur_level + 1, max_level, memo) + _climb_stairs3(cur_level + 2, max_level, memo)
    return memo[cur_level]

# 法四：动态规划
def climbStairs4(n):
    if n <= 2:
        return n
    a, b = 1, 2
    for i in range(n-2):
        a, b = b, a + b
    return b


# 法六：通项公式


# test
n = 10
print "1:", climbStairs(n)
print "2:", climbStairs2(n)
print "3:", climbStairs3(n)
print "4:", climbStairs4(n)