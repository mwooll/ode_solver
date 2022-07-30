#interactive_visualization
import numpy  as np
import pandas as pd

from importlib import reload  

from bokeh.models import Button, Div, TextInput, RadioButtonGroup
from bokeh.models import ColumnDataSource#, HoverTool
from bokeh.plotting import curdoc, figure
from bokeh.layouts import column, row

from solver_main import initialization
from helper_functions import process_string, plot_u, get_heights, get_error, \
    convert_df_to_dict, convert_string_to_int, convert_string_to_float


"""setting up the file for the input functions and loading the base functions"""
file_name   = "functions_file.py"
with open("template.txt", "r") as template:
    content = template.read()
with open(file_name, "w+") as save_file:
    save_file.write(content)
import functions_file as file


parameters  = {"epsilon":    10**(-10),
               "interval":   [0, 1],
               "boundary":   [0, 0],
               "num_nodes":  9,
               "num_points": 101,
               "function_a": file.a,
               "function_f": file.f,
               "function_g": file.g}

calculated  = {"partition": np.linspace(0, 1, 9),
               "solution":  [],
               "dataframe": None,
               "source":    ColumnDataSource(dict()),
               "heights":   []}

plotted     = {"plot_u": None,
               "plot_a": None,
               "plot_f": None,
               "plot_g": None,
               "plot_e": None,
               "plot_E": None}

colours     = {"u": "red", "a": "green", "f": "blue", 
               "g": "cyan", "e": "purple", "E": "lime"}
legend_text = {"u": "solution", "a": "weight function", 
               "f": "target function", "g": "reference function",
               "e": "absolute error", "E": "integral of error"}
plot_labels = ["main plot", "reference plot", "error plot"]


"""functions to handle the number inputs"""
def partitioner(attr, old, new) -> None:
    converted = convert_string_to_int(new, 3)
    partition_div.text = str(converted)
    if isinstance(converted, int):
        parameters["num_nodes"] = converted

def epsiloner(attr, old, new) -> None:
    converted = convert_string_to_float(new, 10**(-15))
    epsilon_div.text = str(converted)
    if isinstance(converted, (float, int)):
        parameters["epsilon"] = converted
        
    
def left_bounder(attr, old, new) -> None:
    converted = convert_string_to_float(new,
                                        minimum=None,
                                        maximum=parameters["interval"][1]-0.01)
    left_bound_div.text = str(converted)
    if isinstance(converted, (float, int)):
        parameters["interval"][0] = converted

def right_bounder(attr, old, new) -> None:
    converted = convert_string_to_float(new,
                                        minimum=parameters["interval"][0]+0.01)
    right_bound_div.text = str(converted)
    if isinstance(converted, (float, int)):
        parameters["interval"][1] = converted

def pointer(attr, old, new) -> None:
    converted = convert_string_to_int(new, 2)
    points_plot_div.text = str(converted)
    if isinstance(converted, int):
        parameters["num_points"] = converted
        if calculated["source"].data:
            update_plot(button_group.active)


"""functions to handle the inputs to determine the functions a, f and g"""
def functioner_a(attr, old, new) -> None:
    function_a_div.text = process_string("a", new, file_name)
    parameters["function_a"] = reload(file).a

def functioner_f(attr, old, new) -> None:
    function_f_div.text = process_string("f", new, file_name)
    parameters["function_f"] = reload(file).f

def functioner_g(attr, old, new) -> None:
    function_g_div.text = process_string("g", new, file_name)
    parameters["function_g"] = reload(file).g

    if calculated["source"].data:
        new_values = file.g(calculated["source"].data["x"])
        calculated["dataframe"]["g"] = new_values

        e, E = get_error(calculated["dataframe"]["u"], 
                         new_values, 
                         parameters["interval"])
        calculated["dataframe"]["e"] = e
        calculated["dataframe"]["E"] = E
        calculated["source"].data = convert_df_to_dict(calculated["dataframe"])
        if button_group.active != 0:
            update_height(button_group.active)


"""main update function"""
def update_function() -> None:
    print("performing calculations")
    update_residuum_div.text = "performing calculations"
    update_button.disabled  = True
    update_button.label     = "computing..."
    perform_the_computations()

def perform_the_computations() -> None:
    calculated["partition"] = np.linspace(parameters["interval"][0],
                                          parameters["interval"][1],
                                          parameters["num_nodes"])

    calculations = initialization(partition = calculated["partition"],
                                  f_func    = parameters["function_f"],
                                  a_func    = parameters["function_a"],
                                  epsilon   = parameters["epsilon"],
                                  interval  = parameters["interval"])
    calculated["solution"]  = calculations[0]

    update_plot(0)
    button_group.active     = 0
    update_button.disabled  = False
    update_button.label     = "perform the computations"
    update_residuum_div(calculations[1], calculations[2])


def update_plot(index) -> None:
    """updating the plot"""
    nodes    = np.linspace(parameters["interval"][0],
                           parameters["interval"][1],
                           parameters["num_points"])
    u_ready  = plot_u(nodes, calculated["partition"], calculated["solution"])

    a_ready  = parameters["function_a"](nodes)
    f_ready  = parameters["function_f"](nodes)
    g_ready  = parameters["function_g"](nodes)

    e_ready, E_ready = get_error(u_ready, g_ready, parameters["interval"])

    df = pd.DataFrame(list(zip(nodes, a_ready, f_ready,
                               u_ready, g_ready, e_ready, E_ready)),
                      columns=["x", "a", "f", "u", "g", "e", "E"])
    calculated["dataframe"] = df

    calculated["source"].data = convert_df_to_dict(calculated["dataframe"])
    select_plot(index)

def plot_it(functions: list) -> None:
    """plotting the new functions"""
    for func in functions:
        line = main_plot.line(x="x", y=func, color=colours[func],
                              line_width=3, source=calculated["source"],
                              legend_label=legend_text[func], name=func)
        plotted[f"plot_{func}"] = line  

def select_plot(new: int) -> None:
    """turn everything previously plotted invisible"""
    for value in plotted.values():
        if value != None:
            value.visible = False
    update_height(new)
    update_interval(parameters["interval"][0], parameters["interval"][1])

    """plotting the new functions"""
    if new == 0:
        plot_it(["a", "f", "u"])
    if new == 1:
        plot_it(["g", "u"])
    if new == 2:
        plot_it(["e", "E"])

def update_height(index: int) -> list:
    """updating the height of the plot"""
    calculated["heights"]   = get_heights(calculated["dataframe"])
    h = (calculated["heights"][index][0] - calculated["heights"][index][1])*0.1

    main_plot.y_range.end   = calculated["heights"][index][0] + h
    main_plot.y_range.start = calculated["heights"][index][1] - h

    return calculated["heights"][index]

def update_interval(start: int=0, end: int=1) -> None:
    """updating the length of the plotted interval"""
    length = end - start
    main_plot.x_range.start = start - 0.1*length
    main_plot.x_range.end   = end   + 0.1*length

def update_residuum_div(residuum, iterations) -> None:
    """updating the div which shows the residuum of the solution"""
    residuum_div.text = (f"Calculation finished with a residuum of {residuum}," +
                         f" after {iterations} iteration" + "s"*(iterations!=1))


"""defining standard sizes"""
height_1 =  50
height_2 =  25
height_3 =  40

width_1  = 425
width_2  = 100
width_3  = 325
width_4  = 230

margin = (30, 0, 0, 0)

"""1. column: description and input management"""
description = ("This is a numerical 2nd order ode solver, to solve:"
               + "</br>" +
               "-(au')' = f in I = ]a, b[ and u = 0 on dI = {a, b}."
               + "</br>" +
               "where a and f can be given through the text inputs below." 
               + "</br>" + "</br>" +
               "If we have a function g in C^2(I), with g = 0 on {a, b}"
               + "</br>" +
               "and set f = -g'' and a = 1  =>  u will be a discretization of g.")
description_div = Div(text=description,
                      height=100, width=width_1)

"""partition"""
partition_title = "length of the partition"
partition_inp   = TextInput(title=partition_title,
                            value="9", height=height_1, width=width_2)
partition_inp.on_change("value", partitioner)
partition_div   = Div(text="9", margin=margin,
                      height=height_2, width=width_3)

"""choose epsilon"""
epsilon_inp     = TextInput(title="error bound", 
                            value="10**(-10)", height=height_1, width=width_2)
epsilon_inp.on_change("value", epsiloner)
epsilon_div     = Div(text="10**(-10)", margin=margin,
                      height=height_2, width=width_3)

"""function a"""
function_a_inp  = TextInput(title="weight function",
                            value="1", height=height_3, width=width_1)
function_a_inp.on_change("value", functioner_a)
function_a_div  = Div(text="a(x) = 1", height=height_2, width=width_1)

"""function f"""
function_f_inp  = TextInput(title="target function",
                            value="1", height=height_3, width=width_1)
function_f_inp.on_change("value", functioner_f)
function_f_div  = Div(text="f(x) = 1", height=height_2, width=width_1)

"""function g"""
function_g_inp  = TextInput(title="reference function",
                            value="0", height=height_3, width=width_1)
function_g_inp.on_change("value", functioner_g)
function_g_div  = Div(text="g(x) = 0", height=height_2, width=width_1)


"""plot update button"""
update_button   = Button(label="perform the computations", 
                         height=height_1, width=width_1)
update_button.on_click(perform_the_computations)


"""residuum div"""
residuum_div    = Div(text="", width=width_1, height=height_1)

"""2. column: plot and plot management"""
main_plot       = figure(title="", plot_height=500, plot_width=1000,
                         x_range=[-0.05, 1.05], y_range=[-1, 1])

"""RadioButtonGroup to select the plot"""
button_group    = RadioButtonGroup(labels=plot_labels, width=1000, height=50)
button_group.on_click(select_plot)

"""TextInput to choose the interval used during the next computation"""
left_bound_inp  = TextInput(title="left interval bound",
                            value="0", width=width_2, height=height_3)
left_bound_inp.on_change("value", left_bounder)
left_bound_div  = Div(text="0", margin=margin,
                      height=height_2, width=width_4)

right_bound_inp = TextInput(title="right interval bound",
                            value="1", width=width_2, height=height_3)
right_bound_inp.on_change("value", right_bounder)
right_bound_div = Div(text="1", margin=margin,
                      height=height_2, width=width_4)

"""TextInput to choose the number of points plotted"""
points_plot_inp = TextInput(title="number of points to plot",
                            value="101", width=width_2, height=height_3)
points_plot_inp.on_change("value", pointer)
points_plot_div = Div(text="101", margin=margin,
                      height=height_2, width=width_4)


"""building the layout and adding it to curdoc()"""
layout = column(row(column(description_div,
                           row(partition_inp,  
                               partition_div),
                           row(epsilon_inp,    
                               epsilon_div),
                           function_a_inp,
                           function_a_div,
                           function_f_inp,
                           function_f_div,
                           function_g_inp,
                           function_g_div),
                    main_plot),
                row(update_button,   button_group),
                row(residuum_div, 
                    left_bound_inp,  left_bound_div,
                    right_bound_inp, right_bound_div,
                    points_plot_inp, points_plot_div))

curdoc().add_root(layout)
curdoc().title = "interactive ode solver"
