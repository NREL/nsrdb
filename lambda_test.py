"""
Lambda function handler tester
"""
from _lambda import handler
from rex import safe_json_load
import sys
import time

if __name__ == '__main__':
    event = safe_json_load(sys.argv[1])
    ts = time.time()
    handler(event, None)
    print('NSRDB lambda runtime: {:.4f} minutes'
          .format((time.time() - ts) / 60))
