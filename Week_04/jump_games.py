# coding:utf-8

class Solution(object):
    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        ep = len(nums) - 1
        for i in range(len(nums)-1,-1,-1):
            if nums[i] + i >= ep: ep = i
        return ep == 0