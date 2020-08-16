# coding:utf-8

# 法一
# S1, 不考虑合法性，只用左右括号填满N个格子
def generateParenthesis(n):
    return _generate(0, 2 * n, "")


def _generate(cur, max, cur_string):
    # terminator
    if cur > max:
        print cur_string
        return
    
    # process cur level
    s1 = cur_string + "("
    s2 = cur_string + ")"
    
    # drill down
    _generate(cur + 1, max, s1)
    _generate(cur + 1, max, s2)
    
    # reverse state


# S2，考虑合法性
def generateParenthesis2(n):
    result = []
    _generate2(0, 0, n, "", result)
    return result


def _generate2(left, right, max_left, cur_string, result):
    # terminator
    # if len(cur_string) == 2 * max_left:
    if left == max_left and right == max_left:
        result.append(cur_string)
        return
    
    # process cur level
    # drill down
    if left < max_left:
        _generate2(left + 1, right, max_left, cur_string + "(", result)
    if right < left:
        _generate2(left, right + 1, max_left, cur_string + ")", result)
    
    # reverse state


# method2
def generateParenthesis3(n):
    result = []
    _generate3(result, "", n, n)
    return result


def _generate3(result, cur_str, left, right):
    if left == 0 and right == 0:
        result.append(cur_str)
    if left != 0: _generate3(result, cur_str + "(", left - 1, right)
    if right != 0 and right > left: _generate3(result, cur_str + ")", left, right - 1)


# test
n = 3
# print "1:", generateParenthesis(n)
print "2:", generateParenthesis2(n)
print "3:", generateParenthesis3(n)
