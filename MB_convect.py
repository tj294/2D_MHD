"""
Dedalus v3 script for 2D MHD simulations in a Cartesian box. 
Usage:
    MB_convect.py [options]

Options:
    --Ra=<Ra>           # Rayleigh number
    --Pr=<Pr>           # Prandtl number [default: 1]
    --Pm=<Pm>           # Magnetic Pr [default: 10]
    --Q=<Q>             # Chandrasekhar number [default: 1e3]
    --Lx=<Lx>           # Aspect Ratio of Box [default: 1]
    --Lz=<Lz>           # Depth of box [default: 1]
    --Nx=<Nx>           # Horizontal resolution [default: 64]
    --Nz=<Nz>           # Vertical resolution [default: 64]
    --maxdt=<maxdt>     # maximum timestep [default: 0.125]
    --stop=<stop>       # Sim stop time [default: 300]
    --top=<top>         # Top Temp BC [default: insulating]
    --bottom=<bottom>   # Bottom temp BC [default: insulating]
    --currie            # run with cos heating function
    --kazemi            # run with exp heating function
    --kinematic         # kinematic dynamo
    --background        # include background field
    --alfven            # include alfven velocity limit
    --slip=<slip>       # Slip conditions (no/free) [default: free]
    -o OUT, --output=<OUT> # Output directory [default: ./data/output/]
"""
import numpy as np
import dedalus.public as d3
import logging
from docopt import docopt
from pathlib import Path

args = docopt(__doc__, version='1.0.0')
logger = logging.getLogger(__name__)



# Parameters
Lx = float(args['--Lx'])
Lz = float(args['--Ly'])
Nx, Nz = int(args['--Nx']), int(args['--Nz'])
Rayleigh = float(args['--Ra'])
Prandtl = float(args['--Pr'])
Pm = float(args['--Pm'])
Q = float(args['--Q'])
stop_sim_time = float(args['--stop'])
max_timestep = float(args['--maxdt'])
if args['--kinematic']:
    kinematic = True
else:
    kinematic = False
if args['--background']:
    include_background_field = True
else:
    include_background_field = False
if args['--alfven']:
    use_alfven_velocity_limit = True
else:
    use_alfven_velocity_limit = False
outpath = Path(args['-o'])

dealias = 3/2
timestepper = d3.RK222
dtype = np.float64

# Bases
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)

# Fields
p = dist.Field(name='p', bases=(xbasis,zbasis))
T = dist.Field(name='T', bases=(xbasis,zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis))
A = dist.Field(name='A', bases=(xbasis,zbasis)) # A_y component of vector potential, 2D 
phi = dist.Field(name='phi', bases=(xbasis,zbasis))




tau_p = dist.Field(name='tau_p')
tau_T1 = dist.Field(name='tau_T1', bases=xbasis)
tau_T2 = dist.Field(name='tau_T2', bases=xbasis)
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xbasis)
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=xbasis)

tau_A1 = dist.Field(name='tau_A1', bases=xbasis)
tau_A2 = dist.Field(name='tau_A2', bases=xbasis)
tau_phi = dist.Field(name='tau_phi')



# Substitutions
kappa = (Rayleigh * Prandtl)**(-1/2)
nu = (Rayleigh / Prandtl)**(-1/2)
#eta = (Pr/(Rayleigh*(Pm**2)))**(1/2)
eta = nu/Pm
Ma_sq = (Rayleigh*Pm/(Q*Prandtl))

x, z = dist.local_grids(xbasis, zbasis)
ex, ez = coords.unit_vector_fields(dist)
lift_basis = zbasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)
grad_u = d3.grad(u) + ez*lift(tau_u1) # First-order reduction
grad_T = d3.grad(T) + ez*lift(tau_T1) # First-order reduction

grad_A = d3.grad(A) + ez*lift(tau_A1)
dx = lambda F: d3.Differentiate(F, coords['x'])
dz = lambda F: d3.Differentiate(F, coords['z']) 
#dphidy = d3.grad(phi) + ez*lift(tau_phi)
#B = dist.VectorField(coords, name='B', bases=(xbasis,zbasis))
w = u@ez
Tz= grad_T@ez
dzux = dz(u@ex)
Bz = dx(A)
Bx = -dz(A)
Jy = -d3.lap(A)
ux = ex@u
uz = ez@u
Btrans = Bz*ex - Bx*ez
Bvec = Bx*ex + Bz*ez
#? Bvec = Bx*ex + Bz*ez # insted of above?
B_scale = (Q*nu*eta*4.0*np.pi)**(1/2) # characteristic scale for B

B_back = B_scale
B_back_vector = B_back*ex + B_back*ez


# Problem

problem = d3.IVP([p, T, u, A, tau_p, tau_T1, tau_T2, tau_u1, tau_u2, tau_A1, tau_A2], namespace=locals())
problem.add_equation("trace(grad_u) + tau_p = 0")
problem.add_equation("dt(T) - kappa*div(grad_T) + lift(tau_T2) = - u@grad(T)")
if (kinematic):
    logger.info("Assuming kinematic MHD, no Lorentz force feedback")
    problem.add_equation("dt(u) - nu*div(grad_u) + grad(p) - T*ez + lift(tau_u2) = - u@grad(u)") 
else : 
    if (include_background_field): 
        logger.info("Including Lorentz force feedback including background field")
        problem.add_equation("dt(u) - nu*div(grad_u) + grad(p) - T*ez + (1/Ma_sq)*Jy*B_back_vector + lift(tau_u2) = - u@grad(u)- (1/Ma_sq)*Jy*Btrans ") 
    else:
        logger.info("Including Lorentz force feedback without background field")
        problem.add_equation("dt(u) - nu*div(grad_u) + grad(p) - T*ez + lift(tau_u2) = - u@grad(u) - (1/Ma_sq)*Jy*Btrans")         

if (include_background_field):
    logger.info("including background field induction term")
    problem.add_equation("dt(A)  - eta*div(grad_A) + ux*B_back + lift(tau_A2) = - u@grad(A)") 
    
else:
    logger.info("ignoring background field induction term")
    problem.add_equation("dt(A)  - eta*div(grad_A) + lift(tau_A2) = - u@grad(A)") 
    

problem.add_equation("T(z=0) = Lz")
problem.add_equation("T(z=Lz) = 0")
if args['--slip'] == 'no':
    problem.add_equation("u(z=0) = 0") # no-slip
    problem.add_equation("u(z=Lz) = 0") # no-slip
elif args['--slip'] == 'free':
    problem.add_equation("w(z=0) = 0")
    problem.add_equation("w(z=Lz) = 0")
    problem.add_equation("dzux(z=0) = 0")
    problem.add_equation("dzux(z=Lz) = 0")
else:
    logger.error(f"Slip condition {args['--slip']} not recognised.\nPlease use 'no' or 'free'.")
    exit(-10)

problem.add_equation("integ(p) = 0") # Pressure gauge
#problem.add_equation("integ(phi) = 0") # Pressure gauge

problem.add_equation("dz(A)(z=Lz) = 0")
problem.add_equation("dz(A)(z=0) = 0")






# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
T.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
T['g'] *= z * (Lz - z) # Damp noise at walls
T['g'] += Lz - z # Add linear background

#A['g'] =1.0*x

# Analysis and snapshots
states = solver.evaluator.add_file_handler(outpath.joinpath('states'), sim_dt=1.0, max_writes=5000)

snapshots = solver.evaluator.add_file_handler(outpath.joinpath('snapshots'), sim_dt=0.25, max_writes=5000)
snapshots.add_task(T, name='buoyancy')
snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')
snapshots.add_task(A, name='A')
snapshots.add_task(-dz(A), name='Bx')
snapshots.add_task(dx(A), name='Bz')


analysis = solver.evaluator.add_file_handler(outpath.joinpath('analysis'), sim_dt=0.25, max_writes=5000)
analysis.add_task(d3.Integrate(T,coords['x'])/Lx, layout='g', name='<T>_x')
analysis.add_task(d3.Integrate(Tz,coords['x'])/Lx, layout='g', name='<Tz>_x')

# Mean Re
analysis.add_task(d3.Average(np.sqrt(u@u)/nu , ('x', 'z')), layout='g', name='Reavg')

# Flux decomposition - Internal energy equation
analysis.add_task(d3.Integrate(T*w,coords['x'])/Lx, layout='g', name='L_conv')

analysis.add_task(-1.0*kappa*(d3.Integrate(Tz, coords['x'])/Lx), layout='g', name='L_cond_tot')

# Nusselt
analysis.add_task( 1.0 + d3.Integrate(T*w, coords)/(kappa*Lx*Lz), layout='g', name='Nusselt')

# magnetic energy
mag_E  = np.sqrt(Bvec@Bvec)
analysis.add_task(d3.Integrate(mag_E,coords['x'])/Lx, layout='g', name='ME')


# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.5, threshold=0.05,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

if (use_alfven_velocity_limit):
    u_alfven_pert = grad_A*B_scale

    CFL.add_velocity(u_alfven_pert)




# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt(u@u)/nu, name='Re')
flow.add_property(np.sqrt(A**2), name='A')



# Main loop
startup_iter = 10
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            max_Re = flow.max('Re')
            max_A = flow.max('A')
            logger.info('Iteration=%i, Time=%e, dt=%e, max(Re)=%f, max(A)=%g' %(solver.iteration, solver.sim_time, timestep, max_Re, max_A))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

