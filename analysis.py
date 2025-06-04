"""
Analysis script for outputs from MB_convect.py

Usage:
    analysis.py <DIREC> [options]

Options:
    -g                      # Create a gif
    -e                      # Plot energy time tracks
    -f                      # Plot fluxes
    -n                      # Calculate Nusselt numbers
    -t                      # Plot temperature profile, store T_cz
    --cadence=<cadence>     # Plot cadence for gif [default: 1]
    --Apert                 # Only plot contours in perturbations to A in gif
    --ASI=<ASI>             # Start Time for averaging [default: -1]
    --AD=<AD>               # Duration to average [default: 200]
    -v                      # Verbose output
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py as h5
from pathlib import Path
from docopt import docopt
import re, shutil, json
import imageio.v2 as iio

args = docopt(__doc__, version='0.1.0')

def get_ave_indices(time, SI, AD):
    if SI<0:
        AEI=-1
        ASI = np.abs(time - (time[-1]-AD)).argmin()
    else:
        ASI = np.abs(time - SI).argmin()
        AEI = np.abs(time - (time[ASI]+AD)).argmin()
    return ASI, AEI

def vprint(text, verbose=args['-v'], end='\n'):
    if verbose:
        print(text, end=end)

def rolling_average(quantity, time, window: float):
    assert len(time) == len(quantity)
    run_ave = []
    for i, t0 in enumerate(time):
        mean = np.nanmean(
            quantity[(time > t0 - window / 2) & (time <= t0 + window / 2)]
        )
        run_ave.append(mean)
    return np.array(run_ave)

vprint(args)


direc = Path(args['<DIREC>'])
outpath = direc.joinpath("images/")
outpath.mkdir(exist_ok=True)
snap_files = sorted(list(direc.glob("snapshots/*.h5")), key=lambda f: int(re.sub(r'\D', '', str(f.name))))
a_files = sorted(list(direc.glob("analysis/*.h5")), key=lambda f: int(re.sub(r'\D', '', str(f.name))))
if len(snap_files)==0:
    print(f"No data files found in {direc}/snapshots/")
    raise FileNotFoundError
if len(a_files)==0:
    print(f"No data files found in {direc}/analysis/")
    raise FileNotFoundError
outdict = {}

with open(direc.joinpath("run_params/runparams.json"), 'r') as f:
    indata = json.load(f)
outdict['Rf'] = indata['Ra']
outdict["Pr"] = indata['Pr']
outdict['Pm'] = indata['Pm']
outdict["Q"] = indata['Q']
outdict['gamma'] = indata['Lx'] / indata['Lz']



if args['-e']:
    vprint("Loading Energies")    
    for i, afile in enumerate(a_files):
        if i==0:
            with h5.File(afile, 'r') as f:
                atime = np.array(f['tasks']['Reavg'].dims[0]['sim_time'])
                Re_inst = np.array(f['tasks']['Reavg'])[:, 0, 0]
                Nu_inst = np.array(f['tasks']['Nusselt'])[:, 0, 0]
                ME_prof = np.array(f['tasks']['ME'])[:, 0, :]
                z = np.array(f['tasks']['ME'].dims[2]['z'])
        else:
            with h5.File(afile, 'r') as f:
                atime = np.concatenate((atime, np.array(f['tasks']['Reavg'].dims[0]['sim_time'])))
                Re_inst = np.concatenate((Re_inst, np.array(f['tasks']['Reavg'])[:, 0, 0]))
                Nu_inst = np.concatenate((Nu_inst, np.array(f['tasks']['Nusselt'])[:, 0, 0]))
                ME_prof = np.concatenate((ME_prof, np.array(f['tasks']['ME'])[:, 0, :]))
            
    aASI, aAEI = get_ave_indices(atime, float(args['--ASI']), float(args['--AD']))
    ME_inst = np.trapezoid(ME_prof, x=z, axis=1)
    vprint("Plotting Energies")
    fig, ax = plt.subplots()
    Re_ave = rolling_average(Re_inst, atime, 50)
    Nu_ave = rolling_average(Nu_inst, atime, 50)
    ME_ave = rolling_average(ME_inst, atime, 50)
    ax.plot(atime, Re_ave, color='k')
    ax.plot(atime, Nu_ave, color='red')
    ax.plot(atime, ME_ave, color='forestgreen')
    ylims = ax.get_ylim()
    print(ylims)
    ax.scatter(atime, Re_inst, label='Re', marker='+', color='C0')
    ax.scatter(atime, Nu_inst, label='Nu', marker='+', color='C1')
    ax.scatter(atime, ME_inst, label="ME", marker='+', color='C2')
    ax.axvspan(atime[aASI], atime[aAEI], color='grey', alpha=0.2)
    ax.set_xlabel(r"$\tau_{ff}$")
    ax.set_ylabel("Energy")
    ax.legend(loc='best')
    ymax = 1.5 * np.max([np.max(Re_inst[aASI:aAEI]), np.max(Nu_inst[aASI:aAEI]), np.max(ME_inst[aASI:aAEI])])
    ax.set_ylim(ylims)
    plt.savefig(outpath.joinpath("energy.pdf"), dpi=500)
    vprint("Done.")

if args['-f']:
    vprint("Loading fluxes")
    for i, afile in enumerate(a_files):
        if i==0:
            with h5.File(afile, 'r') as f:
                atime = np.array(f['tasks']['Reavg'].dims[0]['sim_time'])
                L_conv = np.array(f['tasks']['L_conv'])[:, 0, :]
                L_cond = np.array(f['tasks']['L_cond_tot'])[:, 0, :]
                z = np.array(f['tasks']['L_conv'].dims[2]['z'])
        else:
            with h5.File(afile, 'r') as f:
                atime = np.concatenate((atime, np.array(f['tasks']['Reavg'].dims[0]['sim_time'])))
                L_conv = np.concatenate((L_conv, np.array(f['tasks']['L_conv'])[:, 0, :]))
                L_cond = np.concatenate((L_cond, np.array(f['tasks']['L_cond_tot'])[:, 0, :]))
    aASI, aAEI = get_ave_indices(atime, float(args['--ASI']), float(args['--AD']))
    hwidth = 0.2
    hl = np.abs(z - hwidth).argmin()
    hu = np.abs(z - (z[-1] - hwidth)).argmin()
    lthree = np.abs(z - (z[-1]/3)).argmin()
    uthree = np.abs(z - (2 * z[-1] / 3)).argmin()
    vprint("Plotting Fluxes")
    L_conv_bar = np.nanmean(L_conv[aASI:aAEI], axis=0)
    L_cond_bar = np.nanmean(L_cond[aASI:aAEI], axis=0)
    fig, ax = plt.subplots()
    ax.plot(L_conv_bar, z, c='r', label=r"L$_{conv}$")
    ax.plot(L_cond_bar, z, c='b', label=r'L$_{cond}$')
    ax.plot(L_conv_bar + L_cond_bar, z, c='k', label=r"L$_{tot}$")
    ax.axvline(0, color='grey', ls='--', alpha=0.2)
    ax.axhspan(0, z[hl], color='red', alpha=0.2)
    ax.axhspan(z[hu], z[-1], color='blue', alpha=0.2)
    ax.axhspan(z[lthree], z[uthree], color='grey', alpha=0.2)
    ax.set_xlabel("L")
    ax.set_ylabel("z")
    ax.legend(loc='best')
    plt.savefig(outpath.joinpath("fluxes.pdf"), dpi=500)
    vprint("Done.")


if args['-n']:
    vprint("Calculating Nusselt Numbers")
    for i, afile in enumerate(a_files):
        if i==0:
            with h5.File(afile, 'r') as f:
                atime = np.array(f['tasks']['Reavg'].dims[0]['sim_time'])
                MattNu = np.array(f['tasks']['Nusselt'])[:, 0, 0]
                L_cond = np.array(f['tasks']['L_cond_tot'])[:, 0, :]
                L_conv = np.array(f['tasks']['L_conv'])[:, 0, :]
                z = np.array(f['tasks']['L_conv'].dims[2]['z'])
        else:
            with h5.File(afile, 'r') as f:
                atime = np.concatenate((atime, np.array(f['tasks']['Reavg'].dims[0]['sim_time'])), axis=0)
                MattNu = np.concatenate((MattNu, np.array(f['tasks']['Nusselt'])[:, 0, 0]))
                L_cond = np.concatenate((L_cond, np.array(f['tasks']['L_cond_tot'])[:, 0, :]))
                L_conv = np.concatenate((L_conv, np.array(f['tasks']['L_conv'])[:, 0, :]))
    aASI, aAEI = get_ave_indices(atime, float(args['--ASI']), float(args['--AD']))
    
    hwidth = 0.2
    hl = np.abs(z - hwidth).argmin()
    hu = np.abs(z - (z[-1] - hwidth)).argmin()

    lthree = np.abs(z - (z[-1]/3)).argmin()
    uthree = np.abs(z - (2 * z[-1] / 3)).argmin()

    L_cond_vol = np.trapezoid(L_cond, x=z, axis=1)
    L_conv_vol = np.trapezoid(L_conv, x=z, axis=1)
    Nu_inst = 1 + (L_conv_vol/L_cond_vol)
    print(L_cond.shape)
    print(z.shape)
    print(L_cond[hl:hu].shape)
    print(z[hl:hu].shape)
    L_cond_cz = np.trapezoid(L_cond[:, hl:hu], x=z[hl:hu], axis=1)
    L_conv_cz = np.trapezoid(L_conv[:, hl:hu], x=z[hl:hu], axis=1)
    Nu_cz_inst = 1 + (L_conv_cz / L_cond_cz)
    
    Nu_thirds_inst = 1 + (np.trapezoid(L_conv[:, lthree:uthree]) / np.trapezoid(L_cond[:, lthree:uthree]))

    plt.subplots(figsize=(15,5))
    plt.plot(atime, MattNu, c='blue', label=r"$1 + \langle wT \rangle_V$")
    plt.plot(atime, Nu_thirds_inst, c='purple', label='Mid_third')
    plt.plot(atime, Nu_inst, c='red', label=r'$1 + \frac{\langle L_{conv} \rangle}{\langle L_{cond} \rangle}$')
    plt.plot(atime, Nu_cz_inst, c='green', label=r'$1 + \frac{L_{conv, cz}}{L_{cond, cz}}$')
    # plt.ylim(-100, 100)
    plt.legend(loc='best', ncols=7)
    plt.savefig(outpath.joinpath("Nusselt.pdf"), dpi=500)
    def tave(Nu, ASI=aASI, AEI=aAEI):
        return np.nanmean(Nu[ASI:AEI], axis=0)

    MattNu_ave = np.nanmean(MattNu[aASI:aAEI], axis=0)
    Nu_ave = tave(Nu_inst)
    Nu_cz_ave = tave(Nu_cz_inst)
    Nu_three = tave(Nu_thirds_inst)
    outdict["Nu"] = MattNu_ave
    outdict["Nu_b"] = Nu_ave
    outdict["Nu_cz"] = Nu_cz_ave
    outdict["Nu_third"] = Nu_three
    vprint(f"Matt Nu = {MattNu_ave:.2f}")
    vprint(f"Nu_box = {Nu_ave:.2f}")
    vprint(f"Nu_cz = {Nu_cz_ave:.2f}")
    vprint(f"Nu_third = {Nu_three:.2f}")
    vprint("Done.")

if args['-t']:
    vprint("Calculating temperature profiles")
    for i, afile in enumerate(a_files):
        if i==0:
            with h5.File(afile, 'r') as f:
                atime = np.array(f['tasks']['Reavg'].dims[0]['sim_time'])
                T_prof = np.array(f['tasks']['<T>_x'])[:, 0, :]
                z = np.array(f['tasks']['L_conv'].dims[2]['z'])
        else:
            with h5.File(afile, 'r') as f:
                atime = np.concatenate((atime, np.array(f['tasks']['Reavg'].dims[0]['sim_time'])))
                T_prof = np.concatenate((T_prof, np.array(f['tasks']['<T>_x'])[:, 0, :]))
    
    aASI, aAEI = get_ave_indices(atime, float(args['--ASI']), float(args['--AD']))

    hwidth = 0.2
    hl = np.abs(z - hwidth).argmin()
    hu = np.abs(z - (z[-1] - hwidth)).argmin()

    T_prof_ave = np.nanmean(T_prof[aASI:aAEI], axis=0)
    plt.subplots()
    plt.plot(T_prof_ave, z, color='k')
    plt.axhspan(0, z[hl], color='red', alpha=0.2)
    plt.axhspan(z[hu], z[-1], color='blue', alpha=0.2)
    plt.xlabel('<T>_x')
    plt.ylabel('z')
    plt.savefig(outpath.joinpath('Temp.pdf'))

    T_czl = T_prof_ave[hl]
    T_czu = T_prof_ave[hu]
    dT_cz = T_czl-T_czu
    vprint(f"T_czb: {T_czl:.2f}")
    vprint(f"T_czu: {T_czu:.2f}")
    vprint(f"dT across CZ: {dT_cz:.2f}")
    outdict['dT_cz'] = dT_cz
    outdict['Tl_cz'] = T_czl

    outdict['Tl'] = T_prof_ave[0]
    outdict['dT'] = T_prof_ave[0] - T_prof_ave[-1]
    vprint(f"dT across box: {T_prof_ave[0] - T_prof_ave[-1]:.2f}")
    vprint(f"T(z=0): {T_prof_ave[0]:.2f}")

if args['-g']:
    vprint("Loading snapshots...")
    for i, sfile in enumerate(snap_files):
        if i==0:
            with h5.File(sfile, 'r') as f:
                snap_time = np.array(f['tasks']['buoyancy'].dims[0]['sim_time'])
                x = np.array(f['tasks']['buoyancy'].dims[1]['x'])
                z = np.array(f['tasks']['buoyancy'].dims[2]['z'])
                Temp = np.array(f['tasks']['buoyancy'])
                Bx = np.array(f['tasks']['Bx'])
                Bz = np.array(f['tasks']['Bz'])
                A = np.array(f['tasks']['A1'])
        else:
            with h5.File(sfile, 'r') as f:
                snap_time = np.concatenate((snap_time, np.array(f['tasks']['buoyancy'].dims[0]['sim_time'])))
                Temp = np.concatenate((Temp, np.array(f['tasks']['buoyancy'])), axis=0)
                Bx = np.concatenate((Bx, np.array(f['tasks']['Bx'])), axis=0)
                Bz = np.concatenate((Bz, np.array(f['tasks']['Bz'])), axis=0)
                A = np.concatenate((A, np.array(f['tasks']['A1'])), axis=0)
    X, Z = np.meshgrid(x, z)
    cadence = int(args['--cadence'])
    count = 0
    Bx = lambda A: - np.gradient(A, axis=0)
    Bz = lambda A: np.gradient(A, axis=1)
    if args['--Apert']:
        A_back = 0*X
    else:
        A_back = 1 * X
    fnames = []
    direc.joinpath(f"frames").mkdir(exist_ok=True)
    vprint("Plotting frames...")
    tnorm = mpl.colors.Normalize(vmin=np.min(Temp), vmax=np.max(Temp))
    for tidx, t in enumerate(snap_time):
        if tidx % cadence != 0:
            continue
        vprint(f"\t{t:.1f}/{snap_time[-1]:.1f}", end='\r')
        fig, ax = plt.subplots()
        cf = plt.contourf(X, Z, Temp[tidx].T, 70, cmap='inferno')
        # tsm = plt.cm.ScalarMappable(norm=tnorm, cmap='inferno')
        full_A = A[tidx].T + A_back
        # if count!=0:
            # plt.quiver(X[::10, ::10], Z[::10, ::10], Bx[tidx, ::10, ::10], Bz[tidx, ::10, ::10], color='white')
            # plt.quiver(X[::10, ::10], Z[::10, ::10], Bx(full_A)[::10, ::10], Bz(full_A)[::10, ::10], color='white')
        # norm = mpl.colors.Normalize(vmin=np.min(full_A), vmax=np.max(full_A))
        # norm = mpl.colors.Normalize(vmin=-1, vmax=1)
        cs = plt.contour(X, Z, full_A, 20, colors='white', linestyles='solid')
        # plt.streamplot(x, z, Bx(full_A), Bz(full_A))
        # sm = plt.cm.ScalarMappable(norm=norm, cmap='PiYG')
        # sm.set_array([])
        Tcax = ax.inset_axes([1, 0, 0.05, 1])
        # Bcax = ax.inset_axes([0, 1, 1, 0.05])
        plt.colorbar(cf, cax=Tcax, label='T', pad=0)
        # plt.colorbar(tsm, cax=Tcax, label='T', pad=0)
        # Bcb = plt.colorbar(sm, cax=Bcax, label='A', orientation='horizontal', pad=0)
        # Bcb.ax.xaxis.set_ticks_position('top')
        # Bcb.ax.xaxis.set_label_position('top')
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_title(rf'$\tau_{{ff}}$ = {t:.2f}, R={outdict["Rf"]:.1e}, Q={outdict["Q"]:.2f}')
        fnames.append(direc.joinpath(f'frames/{count:0>3d}.jpg'))
        plt.tight_layout()
        plt.savefig(fnames[-1], dpi=500)
        plt.close()
        count += 1
    vprint("\nWriting Gif")
    if args['--Apert']:
        name = 'pert' 
    else:
        name='heatmap'
    with iio.get_writer(outpath.joinpath(f'{name}.gif'), mode="I") as writer:
        for i, filename in enumerate(fnames):
            vprint(f"\t frame {i+1}/{len(fnames)}", end="\r")
            image = iio.imread(filename)
            writer.append_data(image)
        vprint("\nBuilding gif...")
    vprint("Done.")
    shutil.rmtree(direc.joinpath('frames'))

vprint("Writing outputs.json")
if direc.joinpath("outputs.json").exists():
    with open(direc.joinpath('outputs.json'), 'r') as f:
        exist_dict = json.load(f)
    
    for key in exist_dict.keys():
        if key not in outdict.keys():
            outdict[key] = exist_dict[key]
    

with open(direc.joinpath("outputs.json"), 'w') as f:
    f.write(json.dumps(outdict, indent=4))