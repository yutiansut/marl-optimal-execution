from collections import deque
import numpy as np

### TODO: remaining quantity cannot be computed from the LOB

class ABIDESEnvMetrics():
    def __init__(self, maxlen=5, price_unit = "c") -> None:
        """
        example of msg: {'msg': 'QUERY_SPREAD', 'symbol': 'IBM', 'depth': 500,
        'bids': [(8061, 300)], 'asks': [(8158, 300), (8164, 300)], 
        'data': 8058, 'mkt_closed': False, 'book': ''}
        """
        self.data = {}
        self.price_unit = price_unit
        self.p0 = 0                 # initial price of the day
        self.maxlen = maxlen
        self.__initialized = False

    def getBookCount(self):
        '''
        returns the number of LOB stored in the self.data dictionary
        '''
        return len(self.data[self.getDataAttr()[0]])

    def getDataAttr(self):
        '''
        returns the keys of the self.data dictionary
        '''
        return list(self.data.keys())
    
    def addLOB(self, msg):
        '''
        add content from msg to the self.data dictionary
        '''
        if self.__initialized == False:
            for key in msg:
                self.data[key] = deque(maxlen=self.maxlen)
                self.p0 = msg["data"]
            self.__initialized = True
        
        for key in msg:
            self.data[key].append(msg[key])
    
    def getContentByIndex(self, name=None, idx = 0):
        '''
        name [str]: name of the content to be extracted, if not specified, extract all content with the specified index
        idx [int]: the index of the LOB, 0 is the most resent LOB
        '''
        if idx >= self.getBookCount():
            raise ValueError("LOB with index {} does not exist".format(idx))
        elif name != None:
            return self.data[name][idx]
        else:
            return {key:self.data[key][idx] for key in self.data}
    
    def getVol(self, level=1, idx = 0, mode = "total"):
        '''
        level [int >= 1]: number of levels used to calculate the volumn
        idx [int]: the index of the LOB, 0 is the most resent LOB
        mode [str]: if total, all volume from levels <= specified will be summed, if single, only volume at the level will be returned
        '''
        if idx >= self.getBookCount():
            raise ValueError("LOB with index {} does not exist".format(idx))
        else:
            if mode == "total":
                bidsVol = sum(list(map(lambda x: x[1], self.getContentByIndex("bids", idx=idx)))[:level])
                asksVol = sum(list(map(lambda x: x[1], self.getContentByIndex("asks", idx=idx)))[:level])
            elif mode == "single":
                bidsVol = self.getContentByIndex("bids", idx=idx)[level-1][1]
                asksVol = self.getContentByIndex("asks", idx=idx)[level-1][1]
            else:
                return ValueError("mode is not valid, can only be total or single")
            return bidsVol, asksVol

    def getBidAskPrice(self, level=1, idx=0):
        '''
        level [int >= 1]: the level used to calculate the volume
        idx [int]: the index of the LOB, 0 is the most resent LOB
        '''
        bids = self.getContentByIndex("bids", idx=idx)
        asks = self.getContentByIndex("asks", idx=idx)
        if len(bids) == 0 or len(asks) == 0:
            raise ValueError("Either bids or asks is empty")
        else:
            return bids[level-1][0], asks[level-1][0]

    def getMidPrice(self, level=1, idx = 0):
        '''
        level [int >= 1]: the level used to calculate the mid price
        idx [int]: the index of the LOB, 0 is the most resent LOB
        '''
        bids, asks = self.getBidAskPrice(level=level, idx=idx)
        return (bids + asks)/2

    def getBidAskSpread(self, level=1, idx=0):
        '''
        level [int >= 1]: the level used to calculate the bid-ask spread
        idx [int]: the index of the LOB, 0 is the most resent LOB
        '''
        bids, asks = self.getBidAskPrice(level=level, idx=idx)
        return asks-bids

    def getLogReturn(self, idx = 0):
        '''
        idx [int]: the index of the LOB, 0 is the most resent LOB
        '''
        pt = self.getContentByIndex(name="data", idx = 0)
        return np.log(pt/self.p0)

    def getVolImbalance(self, level=1, idx=0):
        '''
        level [int >= 1]: the level used to calculate the bid-ask volume imbalance
        idx [int]: the index of the LOB, 0 is the most resent LOB
        '''
        bidsVol, asksVol = self.getVol(level=level, idx = idx, mode="total")
        return (bidsVol-asksVol)/(bidsVol+asksVol)

    def getSmartPrice(self, level=1, idx=0):
        '''
        level [int >= 1]: the level used to calculate the smart price
        idx [int]: the index of the LOB, 0 is the most resent LOB
        '''
        bidsPrice, asksPrice = self.getBidAskPrice(level=level, idx=idx)
        bidsVol, asksVol = self.getVol(level=level, idx = idx, mode="single")
        return np.tanh(asksPrice/asksVol - bidsPrice/bidsVol)

    def getMidPriceVolatility(self, level=1):
        '''
        level [int >= 1]: the level used to calculate the smart price
        '''
        MidPriceHist = []
        for idx in range(self.getBookCount()):
            midPrice = self.getMidPrice(level=level, idx = idx)
            MidPriceHist += [np.log(midPrice/self.p0)]
        return np.std(MidPriceHist)


if __name__ == "__main__":
    a = ABIDESEnvMetrics(maxlen=5)

    msg1 = {'msg': 'QUERY_SPREAD', 'symbol': 'IBM', 'depth': 500, 'bids': [(8061, 300)], 'asks': [(8158, 300), (8164, 300), (8189, 300), (8204, 300), (8353, 9)], 'data': 8058, 'mkt_closed': False, 'book': ''}
    msg2 = {'msg': 'QUERY_SPREAD', 'symbol': 'IBM', 'depth': 500, 'bids': [(8061, 300), (8059, 300)], 'asks': [(8158, 1000), (8164, 300), (8189, 300), (8204, 300), (8353, 9)], 'data': 8058, 'mkt_closed': False, 'book': ''}
    msg3 = {'msg': 'QUERY_SPREAD', 'symbol': 'IBM', 'depth': 500, 'bids': [(8061, 500), (8059, 300)], 'asks': [(8158, 1000), (8164, 300), (8189, 300), (8204, 300), (8353, 9)], 'data': 8058, 'mkt_closed': False, 'book': ''}

    for msg in [msg1, msg2]:
        a.addLOB(msg)
    print("latest msg: ", a.getContentByIndex())
    print("Total number of LOB: ", a.getBookCount())
    print("Volume at level 1: ", a.getVol(level=1, mode = "single"))
    print("total Volume up to level 2: ", a.getVol(level=2, mode = "total"))
    print("Mid price: ", a.getMidPrice())
    print("Bid-ask spread: ", a.getBidAskSpread())
    print("Log return: ", a.getLogReturn())
    print("Volumne Imbalance: ", a.getVolImbalance(level=2))
    print("smart price: ", a.getSmartPrice())
    print("mid price volatility: ", a.getMidPriceVolatility())