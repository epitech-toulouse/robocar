import time
import sys

i = 0
while True:
    print(f"Update data {i}")
    sys.stdout.flush() # Important to flush buffer so server gets it immediately
    i += 1
    time.sleep(1)
