# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from collections import defaultdict

def print_table(input_dict : dict, title : str, print_flag : bool = True) -> list:
    """
    Prints a table where keys are column headers and values are items in a row. 
    Returns the table formatted as a list of strings.

    Arguments
    ---------
    input_dict : dict
        A dictionary of keys and values to print out in a table. Values can be dictionaries.
    title : str
        Title of the table, printed before data rows.
    print_flag : bool = True
        An optional boolean value determining whether the generated table is printed.
    """
    col_name_row = "|"
    value_row = "|"
    for k,v in input_dict.items():
        col_len = len(max(str(k), str(v))) + 2
        col_name_row += f"{str(k):^{col_len}}|"
        value_row += f"{str(v):^{col_len}}|"

    break_row = "+{}+".format((len(col_name_row)-2)*'-')
    title_row = f"+{title:^{len(break_row)-2}}+"
    result = [title_row, break_row, col_name_row, break_row, value_row, break_row]
    
    if print_flag:
        print(*result, sep = "\n")
    return result


class Table():
    """
    Accepts and formats an input dictionary in a character table. Stored and printed to standard output.
    
    Arguments
    ---------
    input_dict : dict
        A dictionary of keys and values to print out in a table. Values can be dictionaries.
    title : str
        Optional title to print before table.
    """

    result = []
    sub_result = []
    column_lengths = defaultdict(int)

    class _SubTable():
        """
        Accepts and formats a single, one dimensional dictionary in a character table. Stored and printed to standard output.
        
        Arguments
        ---------
        input_dict : dict
            A dictionary of keys and values to print out in a table.
        title : str
            Optional title to print before table.
        """
        res = []
        def __init__(self, input_dict : dict, title : str = None):
            col_name_row = "|"
            value_row = "|"
            self.title = title
            for k,v in input_dict.items():
                col_len = len(max(str(k), str(v))) + 2
                col_name_row += f"{str(k):^{col_len}}|"
                value_row += f"{str(v):^{col_len}}|"

            break_row = "+{}+".format((len(col_name_row)-2)*'-')
            self.res = [break_row, col_name_row, break_row, value_row, break_row]
            self.title = f"+{self.title:^{len(break_row)-2}}+"
            
        def print(self, print_flag : bool = True) -> None:
            if self.title:
                print(self.title)
            if print_flag:
                print(*self.res, sep = "\n")

    
    def __init__(self, input_dict : dict, title : str = None):
        self.input_dict = input_dict
        self.title = title

        for m in input_dict.keys():
            if self.column_lengths["key"] < len(m):
                self.column_lengths["key"] = max(len(m), len("key")+2) # +2 for header name spacing; less cramped view
            for k,v in input_dict[m].items():
                if isinstance(v, dict):
                    self.column_lengths[k] = None
                else:     
                    update_len = max(len(str(v)),len(k)+2)
                    if self.column_lengths[k] < update_len:
                        self.column_lengths[k] = update_len

        # Formatting header and columns
        col_name_row = "|"
        for k in self.column_lengths.keys(): # Using key order because they shouldn't change while printing
            # if k in self.input_dict and not isinstance(self.input_dict[k], dict):
            if self.column_lengths[k]:
                col_name_row += f"{k:^{self.column_lengths[k]}}|"
        break_row = "+{}+".format((len(col_name_row)-2)*'-')
        self.result = [break_row, col_name_row, break_row]
        self.title = f"+{self.title:^{len(break_row)-2}}+"

        for m in input_dict:
            metric_row = f"|{m:^{self.column_lengths['key']}}|" 
            for k,v in input_dict[m].items():
                # check isinstance v
                if isinstance(v, dict):
                    self.sub_result.append(self._SubTable(v, f"{m} {k}"))
                # else:
                if self.column_lengths[k]:
                    metric_row += f"{str(v):^{self.column_lengths[k]}}|"
            self.result.extend([metric_row, break_row])

    def print(self, print_flag : bool = True) -> None:
        if self.title:
            print(self.title)
        if print_flag:
            print(*self.result, sep = "\n")
        if self.sub_result:
            for t in self.sub_result:
                t.print()

