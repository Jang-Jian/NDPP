from _ctypes_package import runtime_type
from _ndpp_cpython import _cc_logger


def logger(file_name: str, method_name: str, level: runtime_type, 
           message: str, shown: bool = True) -> str:
    """
    logger: Method Logging. \n
    P.S Example. \n
    logger("test.cpp", "test()", ndpp.Info, "testing", true) \n
    Shown on terminal: \n
    [2026-6-6 6:6:6][test.cpp][logging()][RuntimeInfo]: testing.

    :param file_name: Module name.
    :type file_name: str.
    :param method_name: Method name.
    :type method_name: str
    :param level: Log types.
    :type level: ndpp.runtime_type.
    :param message: The information for logging.
    :type message: str.
    :param shown: Whether showing the log or not.
    :type shown: bool.
    :return: The entire log which showing on terminal.
    :rtype: str.
    """
    return _cc_logger(file_name, method_name, level, message, shown)