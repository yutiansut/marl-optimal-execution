from collections import deque

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
            self.__initialized = True
        
        for key in msg:
            self.data[key].append(msg[key])
    
    def getContentByIndex(self, name, idx = 0):
        '''
        idx [int]: the index of the LOB, 0 is the most resent LOB
        '''
        if idx >= self.getBookCount():
            raise ValueError("LOB with index {} does not exist".format(idx))
        else:
            return self.data[name][idx]

    def getMidPrice(self, idx = 0):
        '''
        idx [int]: the index of the LOB, 0 is the most resent LOB
        '''
        ### TODO: consider the case with only bid or ask when the market just opens
        bids = self.getContentByIndex("bids", idx=idx)
        asks = self.getContentByIndex("asks", idx=idx)
        return (bids[0][0] + asks[0][0])/2

    
    def getVol(self, level=1, idx = 0):
        '''
        level [int >= 1]: number of levels used to calculate the volumn
        idx [int]: the index of the LOB, 0 is the most resent LOB
        '''
        if idx >= self.getBookCount():
            raise ValueError("LOB with index {} does not exist".format(idx))
        else:
            bidsVol = sum(list(map(lambda x: x[1], self.getContentByIndex("bids", idx=idx)))[:level])
            asksVol = sum(list(map(lambda x: x[1], self.getContentByIndex("asks", idx=idx)))[:level])
            return bidsVol, asksVol

