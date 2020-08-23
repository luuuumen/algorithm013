# coding:utf-8

class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        return 0 if len(prices) <= 1 else sum([max(prices[i] - prices[i-1],0) for i in range(1,len(prices))])
