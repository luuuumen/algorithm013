# coding:utf-8
from collections import deque
import heapq


class Solution(object):
    # 两数之和
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        
        # dic = dict(zip(nums, range(len(nums))))
        dic = {}
        
        for i, n in enumerate(nums):
            if target - n in dic:
                return [i, dic[target - n]]
            else:
                dic[n] = i
        
        return False
    
    # 树的前-中-后序遍历
    def preorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:
            return []
        return [root.val] + self.preorderTraversal(root.left) + self.preorderTraversal(root.right)  # 前序
        # return self.preorderTraversal(root.left) + [root.val] + self.preorderTraversal(root.right) #中序
        # return self.preorderTraversal(root.left) + self.preorderTraversal(root.right) + [root.val] #后序
    
    # 多叉树的层序遍历
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
    
    # 最小的K个数
    def getLeastNumbers(self, arr, k):
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
    
    # 盛最多的水
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        res, l, r = 0, 0, len(height) - 1
        
        while l < r:
            if height[l] < height[r]:
                res = max(res, height[l] * (r - l))
                l += 1
            else:
                res = max(res, height[r] * (r - l))
                r -= 1
        
        return res
    
    # K个一组反转
    def reverseKGroup(self, head, k):
        cur = head
        count = 0
        while cur and count != k:
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
