# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from collections import defaultdict

def print_table_recursive_save(input_dict : dict, title : str, print_flag : bool = True) -> list:
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
        if isinstance(v, dict): # treat it as separate row
            print_table_recursive(v, f"{title} {k}") # instead of print value, save to a list
        else:
            col_len = len(max(str(k), str(v))) + 2
            col_name_row += f"{str(k):^{col_len}}|"
            value_row += f"{str(v):^{col_len}}|"

    break_row = "+{}+".format((len(col_name_row)-2)*'-')
    title_row = f"+{title:^{len(break_row)-2}}+"
    result = [break_row, title_row, break_row, col_name_row, break_row, value_row, break_row]
    
    if print_flag:
        print(*result, sep = "\n") # pass false to innermost tables
    return result # use list of tables??

def print_table_recursive(input_dict : dict, title : str) -> list:
    table_list = _print_table_recursive_helper([], input_dict, title)
    print("RESULT:\n")
    print(*table_list, sep='\n')
    split_tables = defaultdict(list)

    return table_list

# use helper function to define column widths?
def _print_table_recursive_helper(table_prog : list, input_dict : dict, title : str, key : str = None) -> list:
    """
    Prints a table where keys are column headers and values are items in a row. 
    Returns the table formatted as a list of strings.

    Arguments
    ---------
    table_prog : list
        A list of the table built so far. List of strings, where each string is a printable representation of a row.
    input_dict : dict
        A dictionary of keys and values to print out in a table. Values can be dictionaries.
    title : str
        Title of the table, printed before data rows.
    """

    # Consider placing this in separate helper fcn
    col_name_row = f"| {key} |"
    value_row = f"| {key} |"
    for k,v in input_dict.items():
        if isinstance(v, dict): # treat it as separate row
            _print_table_recursive_helper(table_prog, v, f"{title} {k}", k) # instead of print value, save to a list
        else:
            col_len = len(max(str(k), str(v))) + 2
            col_name_row += f"{str(k):^{col_len}}|"
            value_row += f"{str(v):^{col_len}}|"

    break_row = "+{}+".format((len(col_name_row)-2)*'-')
    title_row = f"+{title:^{len(break_row)-2}}+"
    table_prog.extend([break_row, title_row, break_row, col_name_row, break_row, value_row, break_row])
    
    return table_prog # use list of tables??



def print_table_iterative(input_dict : dict, title : str, print_flag : bool = True) -> list:
    """
    Accepts and formats an input dictionary in a character table. Stored and printed to standard output.
    
    Arguments
    ---------
    input_dict : dict
        A dictionary of keys and values to print out in a table. Values can be dictionaries.
    title : str
        Optional title to print before table.
    print_flag : bool = True
        An optional boolean value determining whether the generated table is printed.
    """
    def sub_print_table_iterative(input_dict : dict, title : str = None, print_flag : bool = True):
        res = []
        col_name_row = "|"
        value_row = "|"
        title = title
        for k,v in input_dict.items():
            col_len = len(max(str(k), str(v))) + 2
            col_name_row += f"{str(k):^{col_len}}|"
            value_row += f"{str(v):^{col_len}}|"

        break_row = "+{}+".format((len(col_name_row)-2)*'-')
        res = [break_row, col_name_row, break_row, value_row, break_row]
        title = f"+{title:^{len(break_row)-2}}+"

        if print_flag:
            if title:
                print(title)
            print(*res, sep = "\n")
        return res

    result = []
    sub_result = []
    column_lengths = defaultdict(int)

    for m in input_dict.keys():
        if column_lengths["key"] < len(m):
            column_lengths["key"] = max(len(m), len("key")+2) # +2 for header name spacing; less cramped view
        for k,v in input_dict[m].items():
            if isinstance(v, dict):
                column_lengths[k] = None
            else:     
                update_len = max(len(str(v)),len(k)+2)
                if column_lengths[k] < update_len:
                    column_lengths[k] = update_len

    # Formatting header and columns
    col_name_row = "|"
    for k in column_lengths.keys(): # Using key order because they shouldn't change while printing
        if column_lengths[k]:
            col_name_row += f"{k:^{column_lengths[k]}}|"
    break_row = "+{}+".format((len(col_name_row)-2)*'-')
    result = [break_row, col_name_row, break_row]
    title = f"+{title:^{len(break_row)-2}}+"

    for m in input_dict:
        metric_row = f"|{m:^{column_lengths['key']}}|" 
        for k,v in input_dict[m].items():
            if isinstance(v, dict):
                sub_result.append(sub_print_table_iterative(v, f"{m} {k}", False))
            if column_lengths[k]:
                metric_row += f"{str(v):^{column_lengths[k]}}|"
        result.extend([metric_row, break_row])

    if print_flag:
        if title:
            print(title)
        print(*result, sep = "\n")
    if sub_result:
        for t in sub_result:
            print(*t, sep = "\n")

