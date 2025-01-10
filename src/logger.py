

import logging
import os
from datetime import datetime


LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(),"logs",LOG_FILE)
os.makedirs(logs_path,exist_ok=True)


LOG_FILE_PATH = os.path.join(logs_path,LOG_FILE)

# saving the log file name. give filename,format of the mesage printed, level of logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
     format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


'''
step to check the logger is started. 
1. give the below line and run the loggers.py file in cmd.
        python src/logger.py
2. We get a file created in the main folder.
'''

# if __name__ == "__main__":
#     logging.info("Logging has started")