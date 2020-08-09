学习笔记

备注：抱歉由于上周请假，两周的总结写一起了。

<a name="eRy97"></a>
# 心得
1.根据场景选择最优数据结构：每种数据结构都有其特点和适用场景，烂熟于心才能在实际工程中熟练适用。比如先进先出想到队列，O(1) 查询想到哈希表，最大最小值用堆等。不存在最优数据结构，需要结合场景选择最合适的，或者做组合。 <br />
<br />2.解题找最小重复单元，再处理边界问题：程序只有顺序／条件／循环这三种运行逻辑。能用程序解的一定能找到最小重复单元，要避免人肉递归／人肉迭代。<br />
<br />3.没思路先上暴力法，再基于此做优化。

4.熟记常用技巧如左右指针、快慢指针、哨兵。<br />
<br />

<a name="aRGSG"></a>
# 概述
| 数据结构 | 特点 | 考点及技巧 |
| --- | --- | --- |
| 数组 | <br />- 连续地址存储。通过存储地址确定前驱和后驱；<br /> | <br />- 考点：数组遍历<br />- 技巧：左右指针、快慢指针<br /> |
| 链表 | <br />- 通过指针确定前驱和后驱，1个结点由val和指针组成；<br />- 特殊：双向链表，循环链表<br /> | <br />- 考点：链表反转／是否有环／入环点<br />- 技巧：双指针、快慢指针<br /> |
| 跳表 | <br />- 多层索引加速链表查询；<br /> | <br />- 理解空间换时间的思想即可<br />- 查询O(logn)，增删O(logn)<br /> |
| 栈 | <br />- LIFO<br />- 特殊：单调栈<br /> | <br />- LIFO用栈<br /> |
| 队列 | <br />- FIFO<br />- 特殊：优先队列、双端队列<br /> | <br />- FIFO用队列<br />- LIFO || FIFO 用双端队列<br />- 极值/其他排序标准用优先队列<br /> |
| 哈希表 | <br />- 根据键值访问存储位置，加速查询<br />- 衍生：map、set<br /> | <br />- O(1)查询用哈希表(/数组)<br /> |
| 树 | <br />- 1前驱多后驱，如文件系统<br />- 特殊：二叉树、二叉搜索树、完全二叉树<br /> | <br />- 遍历：前中后层序遍历<br /> |
| 堆 | <br />- 快速找到最大值／最小值<br />- 特殊：二叉堆<br /> | <br />- 理解：二叉堆实现和操作<br />- 求极值用堆(/优先队列)<br /> |
| 图 | <br />- 多前驱多后驱，由点和边组成；<br />- 树可看作特殊的图，链表可看作特殊的树；<br />- 分类：有向／无向；有权／无权；<br /> | <br />- 遍历：BFS、DFS<br />- 算法：最小路径、最大连通图等<br /> |



<a name="RkQ4A"></a>
# 明细<br />
<a name="C9D1E"></a>
## 1.数组
<a name="c0udK"></a>
### 实现
python的实现是 list<br />查询 O(1)， 增删O(n)<br />

<a name="Oibvc"></a>
### 例题：移动0
```python
class Solution(object):
    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """

        if not nums:
            return []

        p,q = 0,0
        while q < len(nums):
            if nums[q] != 0:
                nums[p] = nums[q]
                if p != q:
                    nums[q] = 0
                p += 1
                q += 1
            else:
                q += 1
```


<a name="Ush7X"></a>
### 例题：盛最多水的容器

- 关键：面积取决于两端，想到用双指针。难点在指针的移动
```python
class Solution(object):
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        res,l,r = 0,0,len(height)-1

        while l < r:
            if height[l] < height[r]:
                res = max(res, height[l] * (r - l))
                l +=  1
            else:
                res = max(res, height[r] * (r-l))
                r -= 1

        return res
```


<a name="uq9E1"></a>
## 2.链表
<a name="ZTfCj"></a>
### 实现
```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None
```
查询O(n), 增删O(1)<br />

<a name="6j0zy"></a>
### 例题：反转链表
```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

# 反转
class Solution(object):
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        pred, cur = None, head

        while cur:
            tmp = cur.next
            cur.next = pred
            pred, cur = cur, tmp

        return pred

# 两两反转-递归
class Solution(object):
    def swapPairs(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head or not head.next:
            return head

        first, second = head, head.next

        first.next = self.swapPairs(second.next)
        second.next = first

        return second

# K个一组反转-递归
class Solution:
    def reverseKGroup(self, head, k) :
        cur = head
        count = 0
        while cur and count!= k:
            cur = cur.next
            count += 1
        if count == k:
            cur = self.reverseKGroup(cur, k)
            while count:
                tmp = head.next
                head.next = cur
                cur = head
                head = tmp
                count -= 1
            head = cur
        return head
```


<a name="E9U1r"></a>
### 例题：链表环

- 技巧：快慢指针
```python

class Solution(object):
    # 是否有环
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        if not head or not head.next:
            return False
        s, f = head, head.next
        while s != f:
            if not f or not f.next:
                return False

            s, f = s.next, f.next.next
        return True

    # 是否有环，有则求环入口
    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        slow, fast = head, head
        while True:
            if not fast or not fast.next: return
            slow, fast = slow.next, fast.next.next
            if slow == fast: break

        fast = head
        while slow != fast:
            slow, fast = slow.next, fast.next
        return fast
```


<a name="Hhz6k"></a>
## 3.栈
<a name="h632j"></a>
### 实现
可基于数组／链表实现，在python中list兼具栈所需的API：

- append：元素入栈 #push in java
- pop：弹出栈顶元素
- lis[-1]：访问栈顶元素，不弹出 #peek in java


<br />查询／插入/删除栈顶O(1)，查询／插入／删除平均O(n)<br />

<a name="SRdlR"></a>
### 例题：有效括号

- 关键：从左向右遍历，遇到 闭括号，右边先return。后进先出用 栈
```python
class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        stack = []

        mapping = {')':'(',']':'[','}':'{'}

        for char in s:
            if char in mapping:
                top = stack.pop() if stack else 'default'

                if mapping[char] != top:
                    return False

            else:
                stack.append(char)

        return not stack
```
<a name="eeau6"></a>
### 例题：柱状图中的最大矩形

- 关键：从左向右遍历，遇到严格大的，右边先return。后进先出用 栈
- 拓展：单调栈、哨兵（通过在头／伪加元素避免边界讨论）
```python
class Solution(object):
    def largestRectangleArea(self, heights):
        """
        :type heights: List[int]
        :rtype: int
        """
        res = 0

        heights = [0] + heights + [0]
        size = len(heights)
        stack = [0]

        for i in range(1,size):
            while heights[i] < heights[stack[-1]]:
                cur_height = heights[stack.pop()]
                res = max(res, cur_height * (i - stack[-1] -1))
            stack.append(i)

        return res
```
<a name="AIOW8"></a>
### 例题：接雨水

- 关键：从左向右遍历，遇到严格大的，右边先return。后进先出用 栈
```python
class Solution(object):
    def trap(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        res = 0
        stack = []

        for i in range(len(height)):
            while stack and height[i] > height[stack[-1]]:
                h = stack.pop()
                if not stack:
                    break
                res += (min(height[i], height[stack[-1]]) - height[h]) \
                       * (i - stack[-1] -1)
            stack.append(i)
        return res
```


<a name="NXl67"></a>
## 4.队列
<a name="uWaLb"></a>
### 实现
1.队列可基于数组／链表实现，核心API类似栈，但操作的是队头。

2.双端队列是值 先进先出 ／ 后进先出，能操作 头部 和 尾部元素。<br />python中的实现是collections.deque，常用API是

- pop
- popleft
- append
- appendleft


<br />3.优先队列是指队列中元素按‘优先度’排，优先度最高的先出。多种实现方式，其中一种实现方式是堆。<br />python中的实现是 queue.PriorityQueue(), 常用API

- put((3,'a')); put((1,'b')) #插入元素, 默认按首个元素的首个子元素排序
- get #优先级最高的元素


<br />4.操作复杂度： 取决于底层实现是数组／链表／堆

<a name="yz9wk"></a>
### 例题：滑动窗口最大值

- 关键：窗口滑动时，一端出一端进，用双端队列
```python
from collections import deque
class Solution(object):
    def maxSlidingWindow(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        deq = deque()
        res = []

        for i, n in enumerate(nums):
            while deq and nums[deq[-1]] < n:
                deq.pop()
            deq += i,

            if deq[0] == i - k:
                deq.popleft()

            if i >= k - 1:
                res.append(nums[deq[0]])

        return res
```


<a name="h9Asu"></a>
## 5.哈希表
<a name="QQK99"></a>
### 实现

1. 原理补充

哈希函数：把键值映射为表中的位置（int)， 如取字符ASCII编码加和 再 模一个数（哈希表的大小）<br />哈希碰撞：由于哈希表太小或哈希函数选的不好，导致 k1,k2 哈希后的 值相同。解决办法是 用 数组+linklist的方式存。<br />

2. 操作

查询O(1), 若哈希碰撞严重则大于O(1)<br />
<br />TODO：HASHMAP源码<br />

<a name="HZdvP"></a>
### 衍生-映射map
特点是k,v存储，其中k不能重复。哈希表是一种特殊的map，原值为k， 哈希后的值为 v<br />python中的实现是 dict<br />

<a name="99lN8"></a>
### 衍生-集合set
特点是无重复元素。在java中hashset是基于hashmap实现的，每次添加新元素，相当于在map中新建k,v（若k存在则替换已有的v），v给一个默认值。<br />python 中的实现是 set<br />

<a name="0ce3a356"></a>
### 例题：两数之和

- 关键：遍历建map，这里巧妙的是建map和判断在一次遍历中完成
```python
class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """

        # dic = dict(zip(nums, range(len(nums))))
        dic = {}

        for i,n in enumerate(nums):
            if target - n in dic:
                return [i,dic[target - n]]
            else:
                dic[n] = i

        return False
```


<a name="oXAcn"></a>
## 6.树
<a name="Dn987"></a>
### 实现
```python
# 二叉树的1个结点
class Node(object):
    def __init__(self, val=None, children=None):
        self.val = val
        self.left = None
        self.right = None

# N叉树的1个结点
class Node(object):
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
```


<a name="Ma2GL"></a>
### 例题：遍历
递归完成前/中/后序遍历
```python
class Solution(object):
    def preorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:
            return []
        return [root.val] + self.preorderTraversal(root.left) + self.preorderTraversal(root.right)  #前序
#        return self.preorderTraversal(root.left) + [root.val] + self.preorderTraversal(root.right) #中序
#        return self.preorderTraversal(root.left) + self.preorderTraversal(root.right) + [root.val] #后序
```

<br />迭代完成N叉树的层序遍历
```python
from collections import deque
class Solution(object):
    def levelOrder(self, root):
        """
        :type root: Node
        :rtype: List[List[int]]
        """
        if not root:
            return []

        result = []
        queue = deque([root])

        while queue:
            layer = []
            for _ in range(len(queue)):
                node = queue.popleft()
                layer.append(node.val)
                queue.extend(node.children)

            result.append(layer)

        return result
```

<br />迭代完成前中后序遍历 #TODO

<a name="nsAHy"></a>
## 7.堆
<a name="HCBpR"></a>
### 实现
堆的实现有多种，如二叉堆、斐波那契堆。其中二叉堆实现简单，但效率不及斐波那契堆。下面介绍二叉堆：<br />

1. 二叉堆特点：

1）完全二叉树；2）根结点 > max(左子，右子）<br />

2. 索引规律

若根节点索引为0，则

- 左子索引为 2i+1
- 右子索引为 2i+2
- 索引i的父结点索引为 floor(i-1)/2

因此二叉堆可用一维数组存储<br />

3. 操作
- 取max/min O(1)
- 插入 O(logn), 过程分2步：
   - S1，插入尾部
   - S2，插入元素向上浮动直到找到所属位置（heapifyUp)
- 删除 max O(logn), 过程分2步：
   - S1，尾部元素替换root
   - S2，尾部元素向下浮动直到找到所属位置（heapifyDown)



4.python实现 heapq<br />heapq.heapify(list) # list to minmum_heap<br />heapq.heappush(heap, ele)<br />heapq.headppop(heap)

#TODO：了解斐波那契等其他堆实现原理

<a name="bJHGl"></a>
### 例题：最小的k个数
输入整数数组 `arr` ，找出其中最小的 `k` 个数。例如，输入4、5、1、6、2、7、3、8这8个数字，则最小的4个数字是1、2、3、4。<br />
<br />用最小堆 O(nlogk)
```python
class Solution:
    def getLeastNumbers(self, arr: List[int], k: int) -> List[int]:
        if k == 0:
            return list()

        hp = [-x for x in arr[:k]]
        heapq.heapify(hp)
        for i in range(k, len(arr)):
            if -hp[0] > arr[i]:
                heapq.heappop(hp)
                heapq.heappush(hp, -arr[i])
        ans = [-x for x in hp]
        return ans
```
其他解法：

- heapq.nsmallest(k, arr)；
- sorted(arr)[:k];
- 快排 O(N) #TODO



<a name="YRSdw"></a>
## 8.图
<a name="7Zq1Q"></a>
### 实现
图的几种常用表示：

1. 邻接矩阵
1. 邻接列表：通过 数组+链表 实现，其中数组包含所有顶点，每个顶点是1个链表头，链表为与该顶点直接相连的边
1. 三元组：(V1,V2,W1) V1，V2为顶点，W1为边的权重。若为有向图，可规定V1，V2的顺序为V1指向V2



<a name="yc2TN"></a>
### 例题：遍历

- 关键：与树的遍历相比，1个顶点可能访问多次，所以要注意记录 visited
```python
# DFS: 用递归
visited = set()

def dfs(node, visited):
    if node in visited:
        return

    visited.add(node)

    process(node)

    neighbors =  generate_neighbors(node)
    for neighbor in neighbors:
        if neighbor not in visited:
            return dfs(neighbor)

# BFS: 用队列，类似树的层序遍历
def BFS(graph, start, end):
    visited = set()

    queue = []
    queue.append([start])

    while queue:
        node = queue.pop()
        visited.add(node)

        process(node)

        nodes = generate_neighbors(node)
        queue.push(nodes)
```

<br />其他题：最短路径、最大连通图 #TODO

