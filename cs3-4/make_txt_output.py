import os
import shutil
import openmdao.api as om
import numpy as np
import dill
import optimization_pyopt as opt
import layout as layout
import matplotlib.pyplot as plt
import niceplots


colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

start_dir = 'cs4_smart_placement_results'

with open(f'{start_dir}/results.pkl', "rb") as dill_file:
    results = dill.load(dill_file)
    
AEP = []
max_AEPs = []
for i, result in enumerate(results):
    AEP.append(result['AEP_final'])
    if i == 0:
        max_AEPs.append(result['AEP_final'])
    else:
        max_AEPs.append(max(result['AEP_final'], np.max(max_AEPs)))
    
AEP = np.array(AEP)
max_AEPs = np.array(max_AEPs)
fcalls = np.arange(len(max_AEPs)) + 1

fcalls_str = ', '.join(str(x) for x in fcalls)
max_AEPs_str = ', '.join(str(x) for x in max_AEPs)

str_list = []
str_list.append("# NREL Optimization Algo History")
str_list.append("# best optimization ID: 11")
str_list.append("# opt. 0 fcalls")
str_list.append(fcalls_str)
str_list.append("# opt. 0 AEP (Whr)")
str_list.append(max_AEPs_str)


import pathlib
dir = 'cs4_full_windrose_good_results'
i_file = 1
for file in pathlib.Path(dir).glob('*.out'):
    if 'SNOPT' not in str(file):
        with open(file, 'r') as f:
            snopt_AEPs = -np.loadtxt(f)
            iterations = np.arange(len(snopt_AEPs)) * 250
            
            fcalls_str = ', '.join(str(x) for x in iterations)
            max_AEPs_str = ', '.join(str(x) for x in snopt_AEPs)
            
            str_list.append(f"# opt. {i_file} fcalls")
            str_list.append(fcalls_str)
            str_list.append(f"# opt. {i_file} AEP (Whr)")
            str_list.append(max_AEPs_str)
            
            i_file += 1


long_str = "\n".join(str_list)

with open("nrel-GAGB-convergence-histories.txt", "w") as text_file:
    text_file.write(long_str)