'''
check the documentation for further details: 
sys:
error_detail.exc_info()  -> it returns 3 values. first & two not nessary. 
                                        3rd value returns which file and which line exception will occur.
'''

import sys 
# from src.logger import logging

# function for error message
def error_message_detail(error,error_detail:sys):
   # see the error message location
    _,_,exc_tb = error_detail.exc_info()
    
    # within the exc_tb we extract the file name by below code => referd in custom exception handling documentation
    file_name = exc_tb.tb_frame.f_code.co_filename
    
    # from the file above we identified. we have locate where the error message line in the below code and print it.
    error_message = "Error occured in Python Script name [{0}] line number [{1}] error message [{2}]".format(
        file_name,exc_tb.tb_lineno.str(error)
    
    )
    return error_message
    
    
class CustomException(Exception):
   
    #call the error message from above function and store in self variable.
    def __init__(self, error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message,error_detail=error_detail)
    
    # give the exception message
    def __str__(self):
        return self.error_message
    
    
    

####code to check the exceptions 
# from src.logger import logging


# if __name__ == "__main__":

#     try:
#         a= 1/0
        
#     except Exception as e:
#         logging.info("Zero Division Error")
#         raise  CustomException(e,sys)

## to run this check file "python src/exception.py"

