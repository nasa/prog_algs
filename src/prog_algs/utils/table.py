# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from collections import defaultdict


class Table():
    result = []
    column_lengths = defaultdict(int)

    def __init__(self, input_dict : dict):
        pass

    def print(self, print_flag : bool = True) -> None:
        if print_flag:
            print(*self.result, sep = "\n")

