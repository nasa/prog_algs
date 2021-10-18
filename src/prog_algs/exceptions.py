# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. All Rights Reserved.


class ProgAlgException(Exception):
    """
    Base Prognostics Model Exception
    """


class ProgAlgInputException(ProgAlgException):
    """
    Prognostics Input Exception - indicates the method input parameters were incorrect
    """


class ProgAlgTypeError(ProgAlgException, TypeError):
    """
    Prognostics Type Error - indicates the model could not be constructed
    """
