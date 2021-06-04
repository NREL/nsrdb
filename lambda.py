"""
Lambda function handler
"""
from cloud_fs import FileSystem
from nsrdb import NSRDB


def handler(event, context):
    """
    Wrapper for NSRDB to allow AWS Lambda invocation

    Parameters
    ----------
    event : dict
        The event dict that contains the parameters sent when the function
        is invoked.
    context : dict
        The context in which the function is called.
    """
    print(f'{event}')
    print(f'{context}')
