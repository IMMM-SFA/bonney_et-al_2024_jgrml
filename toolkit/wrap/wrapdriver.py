import shutil
import os
import subprocess
from datetime import datetime
from os.path import join


class WRAPDriver:
    def __init__(self, wrap_exe_path):
        self.wrap_exe_path = wrap_exe_path
        
    def execute(self, flo_file, execution_folder):
        # run wrap using wrap files in input folder and
        # put output files in output_folder
        # parent_files = os.listdir(parent_folder)
        flo_name = os.path.basename(flo_file).split(".")[0]
        shutil.copyfile(flo_file, join(execution_folder, "C3.FLO"))
        
        # execute wrap
        commandline = f"(echo C3 && echo C3) | wine {self.wrap_exe_path}"
        cmdmsg = subprocess.run(
            commandline, 
            cwd=execution_folder, 
            shell=True, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL)
        # Print periodic status updates
        now = datetime.now()

        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time)
        statusmsg = flo_name + " is done!"
        print(statusmsg)
        os.rename(join(execution_folder, "C3.OUT"), join(execution_folder, f"{flo_name}.OUT"))
        os.rename(join(execution_folder, "C3.MSS"), join(execution_folder, f"{flo_name}.MSS"))
        os.remove(join(execution_folder, "C3.FLO"))