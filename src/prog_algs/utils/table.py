# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from collections import defaultdict


class Table():
    """
    Accepts and formats an input dictionary in a character table. Stored and printed to standard output.
    
    Arguments
    ---------
    input_dict : dict
        A dictionary of keys and values to print out in a table.
    """
    
    result = []
    sub_result = []
    column_lengths = defaultdict(int)

    def __init__(self, input_dict : dict):
        for m in input_dict:
            if self.column_lengths["key"] < len(m):
                self.column_lengths["key"] = max(len(m), len("key")+2) # +2 for header name spacing; less cramped view
            for k,v in input_dict[m].items():
                update_len = max(len(str(v)),len(k)+2)
                if self.column_lengths[k] < update_len:
                    self.column_lengths[k] = update_len

        # Formatting header and columns
        col_name_row = "|"
        for k in self.column_lengths.keys(): # Using key order because they shouldn't change while printing
            col_name_row += f"{k:^{self.column_lengths[k]}}|"
        break_row = "+{}+".format((len(col_name_row)-2)*'-')
        self.result = [break_row, col_name_row, break_row]

        for m in input_dict:
            metric_row = f"|{m:^{self.column_lengths['key']}}|" 
            for k,v in input_dict[m].items():
                metric_row += f"{str(v):^{self.column_lengths[k]}}|"
            self.result.extend([metric_row, break_row])

    def print(self, print_flag : bool = True) -> None:
        if print_flag:
            print(*self.result, sep = "\n")

