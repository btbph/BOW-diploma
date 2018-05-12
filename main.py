import time
from bow import OpenCVBow
# import google_api

start_time = time.time()

N = 27
bow = OpenCVBow(N)
bow.train()

print("--- %s seconds ---" % (time.time() - start_time))
