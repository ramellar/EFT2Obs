import awkward as ak
import vector
from lhereader import LHEReader
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import awkward as ak
import numpy as np
import mplhep
mpl.rcParams['figure.dpi'] = 300

def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    if iteration == total: 
        print()

def lhe_to_awkward(filename):
    reader = LHEReader(filename, weight_mode='dict')

    # Build the event structure as an awkward array
    events = ak.Array([
        {
            "px": [p.px for p in e.particles],
            "py": [p.py for p in e.particles],
            "pz": [p.pz for p in e.particles],
            "E":  [p.energy for p in e.particles],
            "pdgid": [p.pdgid for p in e.particles],
            "status": [p.status for p in e.particles],
            "mass": [p.mass for p in e.particles],
            "weights": e.weights
        }
        for e in reader
    ])

    # Adding a a new field "p4", the 4-vector of the particle in the list events as an awkward array
    vector.register_awkward()
    events = ak.with_field(
        events,
        ak.zip(
            {"px": events.px, "py": events.py, "pz": events.pz, "E": events.E},
            with_name="Momentum4D",
        ),
        "p4"
    )
    
    #Adding a new field with the event number
    events = ak.with_field(events, ak.local_index(events.pdgid, axis=0), "event_id")

    return events

def abs_delta_phi(phi1, phi2):
    dphi = np.abs(phi1 - phi2)
    dphi = ak.where(dphi > np.pi, 2 * np.pi - dphi, dphi)
    return dphi

def delta_phi(phi1, phi2):
    dphi = phi1 - phi2
    dphi = ak.where(dphi > np.pi, 2 * np.pi - dphi, dphi)
    return dphi

def calcDeltaR(part1_eta, part2_eta, part1_phi, part2_phi):

    deta = np.abs (part1_eta - part2_eta)
    dphi = delta_phi(part1_phi, part2_phi)

    deltaR = np.sqrt(deta**2 + dphi**2)
    return deltaR


def plot_awkward_hist(data, weights_dataset ,bins=50, range=[0,10], xlabel="", ylabel="Events", alpha=1, figsize=(10,10), log=False):

    # Flatten in case data is jagged
    flat_data = ak.flatten(data, axis=-1)
    flat_data = np.asarray(flat_data)

    plt.figure(figsize=figsize, dpi=300)

    for label, weights, color in weights_dataset:
        plt.hist(flat_data, bins=bins, range=range, color=color, alpha=alpha,label=label, weights=weights, histtype='step', linewidth=2.5)
        mplhep.cms.label('Private Work', data=False, com = None)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='best')
    if log:
        plt.yscale("log")
    plt.grid(linestyle=':')
    plt.tight_layout()


import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

def plot_awkward_hist_ratio(data, weights_dataset ,bins=50, range=[0,10], xlabel="", ylabel="Events", alpha=0.7, figsize=(10,10), log=False):

    flat_data = ak.flatten(data, axis=-1)
    flat_data = np.asarray(flat_data)
    hep.style.use("CMS")

    # plt.figure(figsize=figsize, dpi=300)

    fig, (ax, rax) = plt.subplots(
        2, 1,
        sharex=True,
        figsize=figsize,
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05}
    )
    hep.cms.label("Private Work", data=False, ax=ax)

    histograms = []
    ref_index = 0  # Index of the reference histogram for ratio plot

    for label, weights, color in weights_dataset:
        # plt.hist(flat_data, bins=bins, range=range, color=color, alpha=alpha,label=label, weights=weights, histtype='step', linewidth=1.5)
        hist, bin_edges = np.histogram(flat_data,bins=bins, range=range, weights=weights)
        hep.histplot(hist, bin_edges,histtype="step",label=label,color=color,ax=ax,alpha=alpha,linewidth=2, yerr=True)
        # hep.histplot(hist, range=range, label=label, ax=ax, histtype='step', weights=weights)
        histograms.append(hist)

    # ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=18)

    if log:
        ax.set_yscale("log")
    ax.grid(linestyle=':')
    
    # --- Ratio plots ---
    # print("histograms", histograms)
    ref_hist = histograms[ref_index]
    # print("ref_hist", ref_hist)
    ref_label = weights_dataset[ref_index][0]

    for i, (label, weights, color) in enumerate(weights_dataset):
        if i == ref_index:
            continue
        ratio = np.divide(histograms[i], ref_hist)
        hep.histplot(
            ratio,
            bin_edges,
            histtype="errorbar",
            label=f"{label}/{ref_label}",
            color=color,
            ax=rax,
            linewidth=2,
        )

    rax.axhline(1.0, color='black', linestyle='--')
    rax.set_ylabel("chg/chg=0")
    rax.set_xlabel(xlabel)
    rax.set_ylim(0.5, 1.5)
    rax.grid(linestyle=':')

    # plt.tight_layout()
    # plt.show()