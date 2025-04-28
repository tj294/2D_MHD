"""
Analysis script for outputs from MB_convect.py

Usage:
    analysis.py <DIREC> [options]

Options:
    -g                      # Create a gif
    -e                      # Plot energy time tracks
    -f                      # Plot fluxes
    -n                      # Calculate Nusselt numbers
    --cadence=<cadence>     # Plot cadence for gif [default: 1]
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

vprint(args)

direc = Path(args['<DIREC>'])
outpath = direc.joinpath("images/")
outpath.mkdir(exist_ok=True)
snap_files = sorted(list(direc.glob("snapshots/*.h5")), key=lambda f: int(re.sub('\D', '', str(f.name))))
a_files = sorted(list(direc.glob("analysis/*.h5")), key=lambda f: int(re.sub('\D', '', str(f.name))))
if len(snap_files)==0:
    print(f"No data files found in {direc}/snapshots/")
    raise FileNotFoundError
if len(a_files)==0:
    print(f"No data files found in {direc}/analysis/")
    raise FileNotFoundError

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
                A = np.array(f['tasks']['A'])
        else:
            with h5.File(sfile, 'r') as f:
                snap_time = np.concatenate((snap_time, np.array(f['tasks']['buoyancy'].dims[0]['sim_time'])))
                Temp = np.concatenate((Temp, np.array(f['tasks']['buoyancy'])), axis=0)
                Bx = np.concatenate((Bx, np.array(f['tasks']['Bx'])), axis=0)
                Bz = np.concatenate((Bz, np.array(f['tasks']['Bz'])), axis=0)
                A = np.concatenate((A, np.array(f['tasks']['A'])), axis=0)
    zz, xx = np.meshgrid(z, x)
    cadence = int(args['--cadence'])
    count = 0
    fnames = []
    direc.joinpath(f"frames").mkdir(exist_ok=True)
    vprint("Plotting frames...")
    for tidx, t in enumerate(snap_time):
        if tidx % cadence != 0:
            continue
        vprint(f"\t{t:.1f}/{snap_time[-1]:.1f}", end='\r')
        fig, ax = plt.subplots()
        cf = plt.contourf(xx, zz, Temp[tidx], 70, cmap='inferno')
        # if count!=0:
            # plt.quiver(xx, zz, Bx[tidx], Bz[tidx], color='white')
        norm = mpl.colors.Normalize(vmin=-np.max(A[tidx]), vmax=np.max(A[tidx]))
        cs = plt.contour(xx, zz, A[tidx], 5, cmap='PiYG', norm=norm)
        sm = plt.cm.ScalarMappable(norm=norm, cmap='PiYG')
        sm.set_array([])
        Tcax = ax.inset_axes([1, 0, 0.05, 1])
        Bcax = ax.inset_axes([0, 1, 1, 0.05])
        plt.colorbar(cf, cax=Tcax, label='T', pad=0)
        Bcb = plt.colorbar(sm, cax=Bcax, label='A', orientation='horizontal', pad=0)
        Bcb.ax.xaxis.set_ticks_position('top')
        Bcb.ax.xaxis.set_label_position('top')
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_title(rf'$\tau_{{ff}}$ = {t:.2f}')
        fnames.append(direc.joinpath(f'frames/{count:0>3d}.jpg'))
        plt.tight_layout()
        plt.savefig(fnames[-1], dpi=500)
        plt.close()
        count += 1
    vprint("\nWriting Gif")
    with iio.get_writer(outpath.joinpath('heatmap.gif'), mode="I") as writer:
        for i, filename in enumerate(fnames):
            vprint(f"\t frame {i+1}/{len(fnames)}", end="\r")
            image = iio.imread(filename)
            writer.append_data(image)
        vprint("\nBuilding gif...")
    vprint("Done.")
    shutil.rmtree(direc.joinpath('frames'))

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

    ME_inst = np.trapz(ME_prof, x=z, axis=1)
    vprint("Plotting Energies")
    fig, ax = plt.subplots()
    ax.scatter(atime, Re_inst, label='Re', marker='+')
    ax.scatter(atime, Nu_inst, label='Nu', marker='+')
    ax.scatter(atime, ME_inst, label="ME", marker='+')
    ax.set_xlabel(r"$\tau_{ff}$")
    ax.set_ylabel("Energy")
    ax.legend(loc='best')
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
    aASI, aAEI = get_ave_indices(atime, float(args['--ASI']), float(args['--AD']))
    vprint("Plotting Fluxes")
    L_conv_bar = np.nanmean(L_conv[aASI:aAEI], axis=0)
    L_cond_bar = np.nanmean(L_cond[aASI:aAEI], axis=0)
    fig, ax = plt.subplots()
    ax.plot(L_conv_bar, z, c='r', label=r"L$_{conv}$")
    ax.plot(L_cond_bar, z, c='b', label=r'L$_{cond}$')
    ax.plot(L_conv_bar + L_cond_bar, z, c='k', label=r"L$_{tot}$")
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
                BBNu_bot = np.array(f['tasks']['BBNu0'])[:, 0, 0]
                BBNu_mid = np.array(f['tasks']['BBNu0.5'])[:, 0, 0]
                BBNu_top = np.array(f['tasks']['BBNu1'])[:, 0, 0]
                WNu_bot = np.array(f['tasks']['WeissNu0'])[:, 0, 0]
                WNu_mid = np.array(f['tasks']['WeissNu0.5'])[:, 0, 0]
                WNu_top = np.array(f['tasks']['WeissNu1'])[:, 0, 0]
        # else:
        #     with h5.File(afile, 'r') as f:
        #         atime = np.concatenate((atime, np.array(f['tasks']['Reavg'].dims[0]['sim_time'])), axis=0)
    aASI, aAEI = get_ave_indices(atime, float(args['--ASI']), float(args['--AD']))
    plt.subplots(figsize=(15,5))
    plt.plot(atime, MattNu, c='blue', label=r"$1 + \langle wT \rangle_V$")

    plt.plot(atime, BBNu_top, c='darkgreen', label=r"$\langle 1-\partial_zT\rangle_H(Lz)$")
    plt.plot(atime, BBNu_bot, c='lawngreen', label=r"$\langle 1-\partial_zT\rangle_H(0)$")
    plt.plot(atime, BBNu_mid, c='lightgreen', label=r"$\langle 1-\partial_zT\rangle_H(0.5)$")

    plt.plot(atime, -WNu_top, c='indianred', label=r"$\langle\partial_zT\rangle_H(Lz)$")
    plt.plot(atime, -WNu_bot, c='red', label=r"$\langle\partial_zT\rangle_H(0)$")
    plt.plot(atime, -WNu_mid, c='salmon', label=r"$\langle\partial_zT\rangle_H(0.5)$")

    # plt.ylim(0, 5)
    plt.legend(loc='best', ncols=7)
    plt.savefig(outpath.joinpath("Nusselt.pdf"), dpi=500)
    def tave(Nu, ASI=aASI, AEI=aAEI):
        return np.nanmean(Nu[ASI:AEI], axis=0)

    MattNu_ave = np.nanmean(MattNu[aASI:aAEI], axis=0)
    vprint(f"Matt Nu = {MattNu_ave:.2f}")

    BBNu_bot_ave = tave(BBNu_bot)
    BBNu_mid_ave = tave(BBNu_mid)
    BBNu_top_ave = tave(BBNu_top)
    vprint("Buckley & Bushby <1 - dz(T)>_H:")
    vprint(f"\tz=0: {BBNu_bot_ave:.2f}")
    vprint(f"\tz=0.5: {BBNu_mid_ave:.2f}")
    vprint(f"\tz=1.0: {BBNu_top_ave:.2f}")

    WNu_bot_ave = tave(WNu_bot)
    WNu_mid_ave = tave(WNu_mid)
    WNu_top_ave = tave(WNu_top)
    vprint("Weiss <dz(T)>_H:")
    vprint(f"\tz=0: {WNu_bot_ave:.2f}")
    vprint(f"\tz=0.5: {WNu_mid_ave:.2f}")
    vprint(f"\tz=1: {WNu_top_ave:.2f}")

    with open(direc.joinpath("run_params/runparams.json"), 'r') as f:
        indata = json.load(f)

    with open(direc.joinpath("outputs.json"), 'w') as f:
        outs = {
            "Rf": indata['Ra'],
            "Pr": indata['Pr'],
            "Pm": indata['Pm'],
            "Q": indata['Q'],
            "gamma": indata['Lx'],
            "Nu": MattNu_ave,
            "BBNu_bot": BBNu_bot_ave,
            "BBNu_mid": BBNu_mid_ave,
            "BBNu_top": BBNu_top_ave,
            "WNu_bot": WNu_bot_ave,
            "WNu_mid": WNu_mid_ave,
            "WNu_top": WNu_top_ave,
        }
        f.write(json.dumps(outs, indent=4))
    vprint("Done.")