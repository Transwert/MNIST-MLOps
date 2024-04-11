import time
from statistics import mean
from model_client import client
import asyncio

class stressTesting:
    def __init__(self):
        """
        NOTE: comment print command in model_client.py
        """
        self.run = client()
    def seqTest(self, numOfReq=1000):
        timeTaken = []
        # run = client()
        for i in range(numOfReq):
            timeStarted = time.time()
            self.run.infer()
            timeTaken.append(time.time()-timeStarted)
        print("Average time taken: ", mean(timeTaken), "for following number of sequential requests: ", numOfReq)

    async def _parallel_test(self, numOfReq=1000):
        """
        as coroutine methods like asyncio.gather don't take functions which give none as response
        this is secondary function
        """
        task = self.run.infer()
        if task == None:
            task = {}
        # return results

    async def paraTest(self, numOfReq=1000):
        """
        Main Function which sends concurrent number of requests,
        By default, does for 1000 requests
        """
        startTime = time.time()
        numberTasks = [self._parallel_test() for i in range(numOfReq)]
        allTasks = await asyncio.gather(*numberTasks)
        endTime = time.time()
        print("Average time taken: ", (endTime - startTime)/numOfReq , "for following concurrent requests: ", numOfReq)



#for Stress Testing of model:
mainFunc = stressTesting()
mainFunc.seqTest()

#for Stress Testing of model:
mainFunc = stressTesting()
asyncio.run(mainFunc.paraTest())


