# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from collections import defaultdict
from typing import Union
from math import log10, ceil

MAX_COLUMN_WIDTH = 5
def set_width(input_value : Union[float, int]) -> str:
    if input_value < (10**MAX_COLUMN_WIDTH):
        ndigits = ceil(log10(input_value))+1
        return f"{input_value:<{ndigits}.{MAX_COLUMN_WIDTH-ndigits}f}"
    else:
        scientific_input = f"{input_value:e}"
        split_e = scientific_input.split("e+")
        num_space = MAX_COLUMN_WIDTH - len(str(split_e[1])) - 2
        split_e[0] = str(split_e[0])[:num_space]
        return f"{split_e[0]}e+{split_e[1]}"
    # what happens if we have 9.999999e+100 but are limited to 5?
    # the exponent itself will occupy 5

def print_table_recursive(input_dict : dict, title : str, print_bool : bool = True) -> defaultdict:
    """
    Prints a table where keys are column headers and values are items in a row. 
    Returns the table formatted as a dictionary of tables represented by a list of str.

    Arguments
    ---------
    input_dict : dict
        A dictionary of keys and values to print out in a table. Values can be dictionaries.
    title : str
        Title of the table, printed before data rows.
    print_flag : bool = True
        An optional boolean value determining whether the generated table is printed.
    """
    row_list = _print_table_recursive_helper([], input_dict, title)[:-7]
    sub_tables = defaultdict(list)
    new_sub_table = []
    for row in row_list:
        new_sub_table.append(row)
        if len(new_sub_table) == 7:
            if sub_tables[len(new_sub_table[0])]:
                sub_tables[len(new_sub_table[0])].extend([new_sub_table[5], new_sub_table[6]])
            else:
                sub_tables[len(new_sub_table[0])].extend(new_sub_table)
            new_sub_table = []

    if print_bool:
        for k in sorted(sub_tables.keys(), reverse=True):
            print(*sub_tables[k], sep='\n')
    return sub_tables

def _print_table_recursive_helper(table_prog : list, input_dict : dict, title : str, key : str = None) -> list:
    """
    Helper function to recursively build subtables as a list of str.

    Arguments
    ---------
    table_prog : list
        A list of the table built so far. List of strings, where each string is a printable representation of a row.
    input_dict : dict
        A dictionary of keys and values to print out in a table. Values can be dictionaries.
    title : str
        Title of the table, printed before data rows.
    key : str = None
        Key for a value row, identifying what event the values belong to.
    """
    col_name_row = "| key |"
    value_row = f"| {str(key):^3} |"
    for k,v in input_dict.items():
        if isinstance(v, dict):
            if key != None:
                to_pass = key
            else:
                to_pass = k
            _print_table_recursive_helper(table_prog, v, f"{title}", to_pass)
        else:
            col_len = len(max(str(k), str(v))) + 2
            col_name_row += f"{str(k):^{col_len}}|"
            value_row += f"{str(v):^{col_len}}|"

    break_row = "+{}+".format((len(col_name_row)-2)*'-')
    title_row = f"+{title:^{len(break_row)-2}}+"
    table_prog.extend([break_row, title_row, break_row, col_name_row, break_row, value_row, break_row])
    
    return table_prog

