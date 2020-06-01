from pyxdsm.XDSM import XDSM

opt = 'Optimization'
func = 'Function'

x = XDSM()

x.add_system('opt1', opt, (r'\text{Gradient-free}', r'\text{optimizer}'))
x.add_system('placement', func, (r'\text{Placement}', r'\text{algorithm}'))
x.add_system('AEP1', func, r'\text{AEP calculation}')
x.add_system('opt2', opt, (r'\text{Gradient-based}', r'\text{optimizer}'))
x.add_system('AEP2', func, r'\text{AEP calculation}')
x.add_system('cons', func, r'\text{Constraint calculation}')

x.connect('AEP1', 'opt1', r'\text{AEP}')
x.connect('opt1', 'opt2', r'\text{Best initial turbine layouts}')

x.connect('opt1', 'placement', (r'\text{Number of turbines}', r'\text{in each region}'))
x.connect('placement', 'AEP1', r'\text{Turbine locations}')
x.connect('opt2', 'AEP2', r'\text{Turbine locations}')
x.connect('opt2', 'cons', r'\text{Turbine locations}')

x.connect('cons', 'opt2', (r'\text{Boundary and distance}', r'\text{constraint values}'))
x.connect('AEP2', 'opt2', r'\text{AEP}')

# x.add_output('opt2', r'\text{Optimal turbine locations}', side='left')

x.add_process(['opt1', 'placement', 'AEP1', 'opt1', 'opt2', 'AEP2', 'cons', 'opt2'], arrow=True)

x.write('xdsm_for_sequential_GA')