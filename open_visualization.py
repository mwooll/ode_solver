#open_visualization
import subprocess

command = "start python -m bokeh serve --show interactive_visualization.py"
process = subprocess.Popen(command, shell=True, start_new_session=True).wait()

# --dev #before "--show"