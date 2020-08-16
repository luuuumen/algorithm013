# coding:utf-8
# 法一：中序遍历，再判断是否递增序列

def validate_bianry_tree(root):
    seq = middle_travel(root)
    return isIncrSeq(seq)


def middle_travel(root):
    if not root:
        return []
    return middle_travel(root.left) + [root.val] + middle_travel(root.right)


def isIncrSeq(seq):
    if not seq or len(seq) == 1:
        return True
    
    q = 1
    while q < len(seq):
        if seq[q - 1] > seq[q]:
            return False
        q += 1
    return True


def isIncrSeq2(seq):
    if not seq or len(seq) == 1:
        return True
    
    for q in range(1, len(seq)):
        if seq[q - 1] > seq[q]:
            return False
    return True


# 法二：递归
def recuse1(root):
    prev = None
    
    def helper(root, prev):
        if not root:
            return True
        if not helper(root.left, prev):
            return False
        if prev and root.val <= prev.val:
            return False
        prev = root
        return helper(root.right, prev)
    
    return helper(root, prev)


# 法三：递归2
def recuse2(root):
    def helper(root, lower, upper):
        if not root: return True
        if root.val <= lower or root.val >= upper: return False
        return helper(root.left, lower, root.val) and helper(root.right, root.val, upper)
    
    return helper(root, -float('inf'), float('inf'))

# 法四：栈 TODO
# https://leetcode.com/problems/validate-binary-search-tree/discuss/32112/Learn-one-iterative-inorder-traversal-apply-it-to-multiple-tree-questions-(Java-Solution)


# helper of test
class node():
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


node1 = node(2)
node2 = node(1)
node3 = node(3)
node1.left = node2
node1.right = node3

# test
# print isIncrSeq([2, 1, 3])
# print isIncrSeq([1, 2, 3])
# print isIncrSeq2([2, 1, 3])
# print isIncrSeq2([1, 2, 3])
# print middle_travel(node1)

print validate_bianry_tree(node1)
print recuse1(node1)
print recuse2(node1)
