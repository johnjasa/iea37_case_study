from pyxdsm.XDSM import XDSM

opt = 'Optimization'
func = 'Function'

x = XDSM()

x.add_system('opt1', opt, (r'\text{Gradient-free}', r'\text{optimizer}'))
x.add_system('placement', func, (r'\text{Placement}', r'\text{algorithm}'))
x.add_system('opt2', opt, (r'\text{Gradient-based}', r'\text{optimizer}'))
x.add_system('AEP', func, r'\text{AEP calculation}')
x.add_system('cons', func, r'\text{Constraint calculation}')

x.connect('opt1', 'placement', (r'\text{Number of turbines}', r'\text{in each region}'))
x.connect('placement', 'opt2', r'\text{Initial turbine locations}')
x.connect('opt2', 'AEP', r'\text{Turbine locations}')
x.connect('opt2', 'cons', r'\text{Turbine locations}')

x.connect('cons', 'opt2', (r'\text{Boundary and distance}', r'\text{constraint values}'))
x.connect('AEP', 'opt2', r'\text{AEP}')
x.connect('opt2', 'opt1', r'\text{Optimal AEP}')

# x.add_output('opt1', r'\text{Optimal turbine locations}', side='right')

x.add_process(['opt1', 'placement', 'opt2', 'AEP', 'cons', 'opt2', 'opt1'], arrow=True)

x.write('xdsm_for_GA')