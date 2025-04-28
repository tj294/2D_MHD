"""
Dedalus v3 script for 2D MHD simulations in a Cartesian box. 
Usage:
    MB_convect.py [options]

Options:
    --Ra=<Ra>           # Rayleigh number
    --Pr=<Pr>           # Prandtl number [default: 1]
    --Pm=<Pm>           # Magnetic Pr [default: 1]
    --Q=<Q>             # Chandrasekhar number [default: 1e3]
    --Lx=<Lx>           # Aspect Ratio of Box [default: 1]
    --Lz=<Lz>           # Depth of box [default: 1]
    --Nx=<Nx>           # Horizontal resolution [default: 64]
    --Nz=<Nz>           # Vertical resolution [default: 64]
    --maxdt=<maxdt>     # maximum timestep [default: 0.125]
    --stop=<stop>       # Sim stop time [default: 300]
    --top=<top>         # Top Temp BC [default: vanishing]
    --bottom=<bottom>   # Bottom temp BC [default: fixed]
    --currie            # run with cos heating function
    --kazemi            # run with exp heating function
    --Delta=<Delta>     # Cosine Heating Delta [default: 0.2]
    --kinematic         # kinematic dynamo
    --background        # include background field
    --alfven            # include alfven velocity limit
    --slip=<slip>       # Slip conditions (no/free) [default: free]
    -i IN, --input=<IN> # Input directory
    -o OUT, --output=<OUT> # Output directory [default: ./data/output/]
"""
import numpy as np
import dedalus.public as d3
import logging, sys, json, re
from datetime import datetime
from docopt import docopt
from pathlib import Path
from mpi4py import MPI

args = docopt(__doc__, version='1.0.0')
logger = logging.getLogger(__name__)

class NaNFlowError(Exception):
    exit_code = -50
    pass

# Parameters
Lx = float(args['--Lx'])
Lz = float(args['--Lz'])
Nx, Nz = int(args['--Nx']), int(args['--Nz'])
Rayleigh = float(args['--Ra'])
Prandtl = float(args['--Pr'])
Pm = float(args['--Pm'])
Q = float(args['--Q'])
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
if args['--currie']:
    Delta = float(args['--Delta'])
    Lz = Lz + 2*Delta
    Lx = Lx*Lz

if args["--input"]:
    restart_path = Path(args["--input"])
    if not restart_path.joinpath('snapshots').exists():
        logger.error(f"Restart Path {restart_path} not found.")
        raise FileNotFoundError
        exit(-10)
    
    logger.info("Reading from {}".format(restart_path))
    with open(restart_path.joinpath('run_params/runparams.json'), 'r') as f:
        inparams = json.load(f)
        Lx = inparams['Lx']
        Lz = inparams['Lz']

outpath = Path(args['--output'])
outpath.mkdir(parents=True, exist_ok=True)
logger.info("Writing to {}".format(outpath))

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
B_scale = (Q*nu*eta*4.0*np.pi)**(1/2) # characteristic scale for B

B_back = B_scale
B_back_vector = B_back*ex + B_back*ez

#? Heating function
F = 1
heat = dist.Field(bases=zbasis)
if args['--currie']:
    heat_func = lambda z: (F/Delta) * (
        1 + np.cos((2 * np.pi * (z - (Delta / 2))) / Delta)
    )
    cool_func = lambda z: (F/Delta) * (
        -1 - np.cos((2 * np.pi * (z - Lz + (Delta / 2))) / Delta)
    )
    heat['g'] = np.piecewise(
        z, [z<=Delta, z >= Lz - Delta], [heat_func, cool_func, 0]
    )
elif args['--kazemi']:
    l = 0.1
    beta = 1
    a = 1 / (0.1 * (1 - np.exp(-1/l)))
    heat_func = lambda z: a * np.exp(-z/l) - beta
    heat['g'] = heat_func(z)
else:
    heat['g'] = np.zeros(heat['g'].shape)

# Problem
problem = d3.IVP([p, T, u, A, tau_p, tau_T1, tau_T2, tau_u1, tau_u2, tau_A1, tau_A2], namespace=locals())
problem.add_equation("trace(grad_u) + tau_p = 0")
problem.add_equation("dt(T) - kappa*div(grad_T) + lift(tau_T2) = - u@grad(T) + heat")
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
    
if args['--bottom']=='fixed':
    problem.add_equation("T(z=0) = Lz")
elif args['--bottom']=='insulating':
    problem.add_equation("Tz(z=0)=0")
elif args['--bottom']=='vanishing':
    problem.add_equation("T(z=0)=0")
else:
    logger.error(f"Bottom BC {args['--bottom']} not recognised.")
    logger.error(f"Accepted values are 'fixed', 'vanishing' or 'insulating'.")
    exit(-1)

if args['--top'] == 'fixed':
    problem.add_equation("T(z=Lz) = Lz")
elif args['--top']=='insulating':
    problem.add_equation("Tz(z=Lz) = 0")
elif args['--top']=='vanishing':
    problem.add_equation("T(z=Lz) = 0")
else:
    logger.error(f"Top BC {args['--top']} not recognised.")
    logger.error(f"Accepted values are 'fixed', 'vanishing' or 'insulating'.")
    exit(-1)

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

# Initial conditions
if args['--input']:
    restart_file = sorted(list(restart_path.glob('states/*.h5')), 
    key=lambda f: int(re.sub('\D', '', str(f.name).rsplit('.')[0])))[-1]
    write, last_dt = solver.load_state(restart_file, -1)
    dt = last_dt
    first_iter = solver.iteration
    if '+' in args['--stop'][0]:
        stop_sim_time = solver.sim_time + float(args['--stop'][1:])
    else:
        stop_sim_time = float(args['--stop'])
    fh_mode = 'append'
else:
    if '+' in args['--stop']:
        stop_sim_time = float(args['--stop'][1:])
    else:
        stop_sim_time = float(args['--stop'])
    
    T.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
    if args['--currie']:
        T['g'] *= z*(Lz-z)
        low_temp = lambda z: (
            (Delta / (4 * np.pi * np.pi))
            * (1 + np.cos((2 * np.pi / Delta) * (z - (Delta / 2))))
            - z * z / (2 * Delta)
            + Lz
            - Delta
        )
        mid_temp = lambda z: F * (-z + Lz - Delta / 2)
        high_temp = lambda z: F * (
            -Delta
            / (4 * np.pi * np.pi)
            * (1 + np.cos((2 * np.pi / Delta) * (z - Lz + Delta / 2)))
            + 1 / (2 * Delta) * (z * z - 2 * Lz * z + Lz * Lz)
        )
        T["g"] += np.piecewise(
            z,
            [z <= Delta, z >= Lz - Delta],
            [low_temp, high_temp, mid_temp],
        )
    else:
        T['g'] *= z * (Lz - z) # Damp noise at walls
        T['g'] += Lz - z # Add linear background
    first_iter = 0
    dt = max_timestep
    fh_mode = 'overwrite'

#A['g'] =1.0*x

# Analysis and snapshots
states = solver.evaluator.add_file_handler(outpath.joinpath('states'), sim_dt=2.0, max_writes=5000, mode=fh_mode)
states.add_tasks(solver.state, layout='g')

snapshots = solver.evaluator.add_file_handler(outpath.joinpath('snapshots'), sim_dt=0.5, max_writes=5000, mode=fh_mode)
snapshots.add_task(T, name='buoyancy')
snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')
snapshots.add_task(A, name='A')
snapshots.add_task(-dz(A), name='Bx')
snapshots.add_task(dx(A), name='Bz')


analysis = solver.evaluator.add_file_handler(outpath.joinpath('analysis'), sim_dt=0.5, max_writes=5000, mode=fh_mode)
analysis.add_task(d3.Integrate(T,coords['x'])/Lx, layout='g', name='<T>_x')
analysis.add_task(d3.Integrate(Tz,coords['x'])/Lx, layout='g', name='<Tz>_x')

# Mean Re
analysis.add_task(d3.Average(np.sqrt(u@u)/nu , ('x', 'z')), layout='g', name='Reavg')


# Flux decomposition - Internal energy equation
analysis.add_task(d3.Integrate(T*w,coords['x'])/Lx, layout='g', name='L_conv')
analysis.add_task(-1.0*kappa*(d3.Integrate(Tz, coords['x'])/Lx), layout='g', name='L_cond_tot')

# magnetic energy
mag_E  = np.sqrt(Bvec@Bvec)
analysis.add_task(d3.Integrate(mag_E,coords['x'])/Lx, layout='g', name='ME')
analysis.add_task(d3.Average(mag_E, coords)/(Lx*Lz), layout='g', name="B^2")

# Nusselt
analysis.add_task( 1.0 + d3.Integrate(T*w, coords)/(kappa*Lx*Lz), layout='g', name='Nusselt')

analysis.add_task(1/Lx * d3.Integrate(Tz(z=0), coords['x']), layout='g', name='WeissNu0')
analysis.add_task(1/Lx * d3.Integrate(Tz(z=0.5), coords['x']), layout='g', name='WeissNu0.5')
analysis.add_task(1/Lx * d3.Integrate(Tz(z=Lz), coords['x']), layout='g', name='WeissNu1')

analysis.add_task(1/Lx * d3.Integrate(1 - Tz(z=0), coords['x']), layout='g', name='BBNu0')
analysis.add_task(1/Lx * d3.Integrate(1 - Tz(z=0.5), coords['x']), layout='g', name='BBNu0.5')
analysis.add_task(1/Lx * d3.Integrate(1 - Tz(z=Lz), coords['x']), layout='g', name='BBNu1')

outpath.joinpath("run_params").mkdir(parents=True, exist_ok=True)
run_params = {
    "Lx": Lx,
    "Lz": Lz,
    "Nx": Nx,
    "Nz": Nz,
    "Ra": Rayleigh,
    "Pr": Prandtl,
    "Pm": Pm,
    "Q": Q,
}
run_params = json.dumps(run_params, indent=4)

with open(outpath.joinpath('run_params/runparams.json'), "w") as run_file:
    run_file.write(run_params)

with open(outpath.joinpath("run_params/args.txt"), "a+") as file:
    if MPI.COMM_WORLD.rank == 0:
        today = datetime.today().strftime("%Y-%m-%d %H:%M:%S\n\t")
        file.write(today)
        file.write("python3 " + " ".join(sys.argv) + "\n")



# CFL
CFL = d3.CFL(solver, initial_dt=dt, cadence=10, safety=0.5, threshold=0.05,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

if (use_alfven_velocity_limit):
    u_alfven_pert = grad_A*B_scale

    CFL.add_velocity(u_alfven_pert)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt(u@u)/nu, name='Re')
flow.add_property(np.sqrt(A**2), name='A')

solver.stop_sim_time = stop_sim_time
solver.warmup_iterations = solver.iteration+100

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            max_Re = flow.max('Re')
            max_A = flow.max('A')
            if np.isnan(max_Re):
                raise NaNFlowError
            logger.info('Iteration=%i, Time=%e, dt=%e, max(Re)=%f, max(A)=%g' %(solver.iteration, solver.sim_time, timestep, max_Re, max_A))
except NaNFlowError:
    logger.error("max Re is nan. Ending loop.")
except:
    logger.error('Exception raised, triggering end of main loop.')
finally:
    solver.evaluate_handlers(dt=timestep)
    solver.log_stats()
    logger.info("Written to {}".format(outpath))

