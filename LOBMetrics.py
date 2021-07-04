from collections import deque

class LOBMetrics:
    def __init__(self) -> None:
        """
        example of msg: {'msg': 'QUERY_SPREAD', 'symbol': 'IBM', 'depth': 500,
        'bids': [(8061, 300)], 'asks': [(8158, 300), (8164, 300)], 
        'data': 8058, 'mkt_closed': False, 'book': ''}
        """

        self.ticker = None 
        self.depth = None
        self.orderBook = deque(maxlen=5)
        self.mktClosed = deque(maxlen=5)
        self.data = deque(maxlen=5)
        self.book = deque(maxlen=5)
        self.bookCount = 0 
    
    def addLOB(self, msg):
        self.tick = msg['symbol']
        self.depth = msg['depth']
        self.orderBook.append((msg['bids'],msg['asks']))
        self.mktClosed.append(msg['mkt_closed'])
        self.data.append(msg['data'])
        self.book.append(msg['data'])
        self.bookCount += 1 
        self.bookCount = min(self.bookCount, 5)
    
    def getLOB(self, idx = 0):
        if idx >= self.bookCount:
            raise ValueError("Invalid index {} in LOB".format(idx))
        else:
            return self.orderBook[idx]
    
    def getData(self, idx = 0):
        if idx >= self.bookCount:
            raise ValueError("Invalid index {} in LOB".format(idx))
        else:
            return self.data[idx]
    
    def getVol(self, idx = 0):
        if idx >= self.bookCount:
            raise ValueError("Invalid index {} in LOB".format(idx))
        else:
            return (self.orderBook[idx][0][1], self.orderBook[idx][1][1])

