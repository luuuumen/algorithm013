学习笔记

模版大法好，但需要通过实践去理解每一步。比如：
- 泛型递归中，怎么设终止条件？怎么设参数？每层具体做什么？参数如何更新？什么情况下需要reverse state?
- 分治中，怎么分解问题？子问题的结果如何merge?

目前通过几个例题有了by case的经验，但拿到新问题时思路还是不清晰，还需要多做题多看题解。

PS.如果老师有其他帮助理解的资料推荐，就更感谢了。

```
def recursion(level, MAX_LEVEL, params):
    # terminator
    if level > MAX_LEVEL:
        return

    # process cur level
    process(level, params)

    # drill down
    recursion(level+1, new_params)

    # reverse state



def devide_conqure(problem, p1, p2):
    # terminator
    if not problem:
        return

    # split cur problem
    subproblems = split_problem(problem)  # 1st important

    # conquer sub problem
    subresult0 = devide_conqure(subproblems[0], p1, p2)
    subresult1 = devide_conqure(subproblems[1], p1, p2)
    # merge sub results
    result = process(subresult0, subresult1)

    # reverse state
    return result
```