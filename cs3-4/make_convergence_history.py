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
    
snopt_AEPs = -np.array([-2.8695576E+06, 
            -2.8711119E+06, 
            -2.8774592E+06, 
            -2.8898599E+06, 
            -2.8899652E+06, 
            -2.8927975E+06, 
            -2.8946043E+06, 
            -2.8969150E+06, 
            -2.8984069E+06, 
            -2.8996422E+06, 
            -2.9001783E+06, 
            -2.9017755E+06, 
            -2.9026302E+06, 
            -2.9034310E+06, 
            -2.9048010E+06, 
            -2.9051996E+06, 
            -2.9055466E+06, 
            -2.9056001E+06, 
            -2.9056575E+06, 
            -2.9059810E+06, 
            -2.9061655E+06, 
            -2.9066533E+06, 
            -2.9067767E+06, 
            -2.9068499E+06, 
            -2.9069054E+06, 
            -2.9069321E+06, 
            -2.9069652E+06, 
            -2.9069980E+06, 
            -2.9070523E+06, 
            -2.9070571E+06, 
            -2.9071078E+06, 
            -2.9071145E+06, 
            -2.9071216E+06, 
            -2.9071587E+06, 
            -2.9071965E+06, 
            -2.9072263E+06, 
            -2.9072383E+06, 
            -2.9072544E+06, 
            -2.9072750E+06, 
            -2.9072770E+06, 
            -2.9072994E+06, 
            -2.9072994E+06, 
            -2.9073064E+06, 
            -2.9073079E+06, 
            -2.9073138E+06, 
            -2.9073183E+06, 
            -2.9072175E+06, 
            -2.9072235E+06, 
            -2.9073442E+06, 
            -2.9072043E+06, 
            -2.9073037E+06, 
            -2.9073575E+06, 
            -2.9072764E+06, 
            -2.9073255E+06, 
            -2.9073609E+06, 
            -2.9073406E+06, 
            -2.9073468E+06, 
            -2.9073731E+06, 
            -2.9073789E+06, 
            -2.9073359E+06, 
            -2.9073583E+06, 
            -2.9073871E+06, 
            -2.9073872E+06, 
            -2.9073927E+06, 
            -2.9070823E+06, 
            -2.9073968E+06, 
            -2.9073929E+06, 
            -2.9074171E+06, 
            -2.9074260E+06, 
            -2.9074339E+06, 
            -2.9074386E+06, 
            -2.9074408E+06, 
            -2.9074425E+06, 
            -2.9074429E+06, 
            -2.9074380E+06, 
            -2.9074380E+06, 
            -2.9074504E+06, 
            -2.9074416E+06, 
            -2.9074744E+06, 
            -2.9074837E+06, 
            -2.9074879E+06, 
            -2.9074932E+06, 
            -2.9074986E+06, 
            -2.9075040E+06, 
            -2.9075090E+06, 
            -2.9075116E+06, 
            -2.9075141E+06, 
            -2.9075149E+06, 
            -2.9075068E+06, 
            -2.9075190E+06, 
            -2.9075224E+06, 
            -2.9075293E+06, 
            -2.9075369E+06, 
            -2.9075390E+06, 
            -2.9075392E+06, 
            -2.9075395E+06, 
            -2.9075395E+06, 
            -2.9075397E+06, 
            -2.9075401E+06, 
            -2.9075401E+06, 
            -2.9075401E+06]) 

max_snopt_AEPs = []
for i, AEP in enumerate(snopt_AEPs):
    if i==0:
        max_snopt_AEPs.append(AEP)
    else:
        max_snopt_AEPs.append(max(AEP, np.max(max_snopt_AEPs)))
max_snopt_AEPs = np.array(max_snopt_AEPs)

max_snopt_AEPs *= 1e-3
max_AEPs *= 1e-3

plt.figure(figsize=(8, 4))

iterations = np.arange(len(max_AEPs))
plt.plot(iterations, max_AEPs, color=colors[0], lw=2)

max_snopt_AEPs = np.repeat(max_snopt_AEPs, 250)
iterations = np.arange(len(max_snopt_AEPs)) + len(max_AEPs)

plt.plot(iterations, max_snopt_AEPs, color=colors[1], lw=2)

plt.ylabel('AEP, GWh', fontsize=14)
plt.xlabel('Function calls', fontsize=14)

ax = plt.gca()

ax.text(1000., 2.84e3, 'GA turbine placement', fontsize=14, color=colors[0])
ax.text(5000., 2.89e3, 'GB fine-tuning', fontsize=14, color=colors[1])

niceplots.adjust_spines()

plt.ylim([2.8e3, 2.91e3])
plt.yticks([np.min(max_AEPs), np.max(max_AEPs), np.max(max_snopt_AEPs)], fontsize=14)
plt.xticks([0., np.min(iterations), np.max(iterations)], fontsize=14)

plt.tight_layout()

plt.savefig('nrel_convergence_history.pdf')