class ProgAlgException(Exception):
    """
    Base Prognostics Model Exception
    """
    pass

class ProgAlgInputException(ProgAlgException):
    """
    Prognostics Input Exception - indicates the method input parameters were incorrect
    """
    pass

class ProgAlgTypeError(ProgAlgException, TypeError):
    """
    Prognostics Type Error - indicates the model could not be constructed
    """
    pass