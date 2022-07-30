#helper_functions
import numpy  as np
import pandas as pd

from solver_main import hut_basis_vector

def process_string(name: str, string: str, file_name) -> (str, str):
    """this function converts the input for a, f, g into an executable function"""

    """pre-processing to lower cases needed to ckeck"""
    string   = string.lower()
    string   = string.replace(" ", "")

    """checking if the input is 'empty'"""
    if string == "" or string == "none":
        if name in ["a", "f"]:
            string = "1"
        if name == "g":
            string = "0"

    """replacing basic mathematical functions and constants with single letters"""
    to_translate = {"pi": "N", "e": "E",
                    "cos": "C", "sin": "S", "tan": "T",
                    "sqrt": "Q", "abs": "A",
                    "exp": "P", "ln": "L", "log": "L",
                    "x": "X", 
                    "^": "**", "+": " + ", "-": " - "}
    
    for key in to_translate.keys():
        string = string.replace(key, to_translate[key])
    
    """removing leftover letters"""
    for symbol in string:
        if symbol.isalpha() == True:
            if symbol not in to_translate.values():
                string = string.replace(symbol, "")

    """re-replacing capital letters with their corresponding function or constant"""
    re_translate = {"N": "pi", "E": "e",
                    "C": "cos", "S": "sin", "T": "tan",
                    "Q": "sqrt", "A": "abs",
                    "P": "exp", "L": "log"}

    function = string
    for key in re_translate.keys():
        string   = string.replace(key, re_translate[key])
        function = function.replace(key, "np."+re_translate[key])

    try:
        eval(function.replace("X", "(0.5)"))
        write_lambda_to_file(name, function, file_name)
        string = f"{name}(x) = {string}".lower()
    except Exception as e:
        print(e)
        return "input is not a recognised function"

    return string

def write_lambda_to_file(usage: str, string: str, file_name: str) -> str:
    to_write = []
    with open(file_name, "r") as read:
        for line in read.readlines():
            if f"def {usage}" in line:
                old_line = line
                index    = len(to_write)
            to_write.append(line)

    if "X" in string:
        to_write[index+1] = "\n"
        to_write[index+2] = "\n"
        to_write[index+3] = f"    return {string}\n"
    else:
        to_write[index+1] = '    if hasattr(X, "__iter__"):\n'
        to_write[index+2] = f"        return np.array([{usage}(x) for x in X])\n"
        to_write[index+3] = f"    return {string}\n"

    with open(file_name, "w") as write:
        for line in to_write:
            write.write(line)
        
    return old_line

def plot_u(X: float or np.array, partition: np.array, solution: np.array) -> float or np.array:
    length  = len(partition)
    to_plot = np.zeros(len(X))

    for k in range(1, length-1):
        to_plot += solution[k-1]*hut_basis_vector(partition, X, k)
    return to_plot


def get_heights(df: pd.DataFrame) -> np.array:
    maximum = df.max()
    minimum = df.min()
    heights = []
    heights.append([np.max(maximum[1:4]), np.min(minimum[1:4])])
    heights.append([np.max(maximum[3:5]), np.min(minimum[3:5])])
    heights.append([np.max(maximum[5:7]), np.min(minimum[5:7])])
    return heights

def get_error(sol: np.array, ref: np.array, interval: list) -> (np.array, np.array):
    width    = interval[1] - interval[0]
    length   = len(sol)
    error    = abs(sol - ref)
    integral = np.array([sum(error[:k+1])/(length-1)*width for k in range(length)])
    return error, integral

def convert_df_to_dict(df: pd.DataFrame) -> dict:
    dic = {}
    for col in df.columns:
        dic[col] = df[col]
    return dic

def convert_string_to_int(string: str, minimum, maximum=None) -> int or str:
    for symbol in string:
        if symbol.isalpha() == True:
            string = string.replace(symbol, "")

    try:
        result = eval(string)
        assert result == int(result)
    except Exception as e:
        print(e)
        return "input is not an integer"

    if minimum != None:
        if result < minimum:
            return f"input must be greater or equal to {minimum}"

    if maximum != None:
        if result > maximum:
            return f"input must be less or equal to {maximum}"

    return result

def convert_string_to_float(string: str, minimum, maximum=None) -> int or str:
    for symbol in string:
        if symbol.isalpha() == True:
            string = string.replace(symbol, "")
    string.replace("^", "**")

    try:
        result = eval(string)
        assert isinstance(result, (float, int))
    except Exception as e:
        print(e)
        return "input is not an number"

    if minimum != None:
        if result < minimum:
            return f"input must be greater or equal to {minimum}"

    if maximum != None:
        if result > maximum:
            return f"input must be less or equal to {maximum}"

    return result