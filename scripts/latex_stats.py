
root = "./"
import os

import numpy as np

from pandas import read_csv

folders = [folder for folder in os.listdir(root) if os.path.isdir(folder)]

print("\\begin{center}")
print("\\begin{tabular}{ c c c c c c c c}")
print("variant & augmentation & mean accuracy & std accuracy  & mean deep fool & std deep fool & blurred acc & blurred std\\\\")

for folder in folders:

    files = os.listdir(root + folder)

    cacc = []
    closs = []
    cdf = []

    bcacc = []
    bcloss = []

    nacc = []
    nloss = []
    ndf = []

    bnacc = []
    bnloss = []

    for f in files:

        d = read_csv(root + folder + "/"+ f)


        if "augment꞉cifar10" in  f:
            cacc.append(d["accuracy"][1])
            cdf.append(d["deep_fool"][1])

            bcacc.append(d["accuracy"][2])
        else:
            nacc.append(d["accuracy"][1])
            ndf.append(d["deep_fool"][1])

            bnacc.append(d["accuracy"][2])
    print(f'{folder} & none & ${np.mean(nacc):.4f}$ & ${np.std(nacc):.4f}$ & ${np.mean(ndf):.4f}$ & ${np.std(ndf):.4f}$ & ${np.mean(bnacc):.4f}$ & ${np.std(bnacc):.4f}$ \\\\')
    print(f'{folder} & cifar10 & ${np.mean(cacc):.4f}$ & ${np.std(cacc):.4f}$ & ${np.mean(cdf):.4f}$ & ${np.std(cdf):.4f}$ & ${np.mean(bcacc):.4f}$ & ${np.std(bcacc):.4f}$\\\\')

print("\\end{tabular}\n\\end{center}")



