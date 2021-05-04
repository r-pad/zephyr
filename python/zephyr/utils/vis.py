import numpy as np
import matplotlib.pyplot as plt

def plotImages(imgs, titles=None, kpts=None, axes = None):
    if axes is None:
        n = len(imgs)
        fig, axes = plt.subplots(1, n, dpi=300, figsize=(2*n, 2))
    else:
        fig = None
        if type(axes) is np.ndarray:
            axes.reshape(-1)
        else:
            axes = np.array([axes])
        n = min(len(imgs), len(imgs))
        
    for i in range(n):
        ax = axes[i]
        ax.imshow(imgs[i])
        ax.axis('off')
        if titles is not None:
            ax.set_title(titles[i], fontsize=6)

        if kpts is not None:
            for kpt in kpts[i]:
                # kpt = [10, 60]
                c = plt.Circle((kpt[1], kpt[0]), 1, color='r', fc='r')
                # c = plt.Circle((10, 60), 1, color='r', fc='r')
                ax.add_patch(c)
        
    return fig, axes