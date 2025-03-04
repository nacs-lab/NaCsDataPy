import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import utils


def plot_images(imgs, sites=None, box_size=None, box_label=None, titles=None):
    # imgs is assumed to be 3 dimension, where the last dimension holds the various images
    # sites is a list of lists or numpy arrays, where we have one site list for each image.
    # An empty site list will not draw any site boxes. Each site should have two coordinates
    # box_size is a list of box sizes for each image.
    # TODO: Fix box size
    nimgs = imgs.shape[2]
    frame_y = imgs.shape[0] #number of rows
    frame_x = imgs.shape[1] #number of columns
    fig, ax = plt.subplots(nimgs, 1, squeeze=False)
    for idx in range(nimgs):
        ax[idx, 0].imshow(imgs[:,:,idx], extent=(-frame_x / 2 + 0.5, frame_x /2 + 0.5, frame_y /2 - 0.5, -frame_y/2 - 0.5), cmap="Greys_r")
        if sites is not None:
            if utils.is_number(box_size):
                this_box_size = box_size
            else:
                this_box_size = box_size[idx]
            these_sites = sites[idx]
            rad = np.ceil((this_box_size-1)/2) # Following the convention in MATLAB...
            for site_idx, site in enumerate(these_sites):
                rect = patches.Rectangle((site[0] - rad - 0.5, site[1] - rad - 0.5), 2 * rad + 1, 2 * rad + 1, edgecolor='red', facecolor='none', linewidth=2)
                ax[idx, 0].add_patch(rect)
                if box_label is not None:
                    if len(box_label[idx]) != 0:
                        this_label = box_label[idx][site_idx]
                    else:
                        this_label = str(site_idx + 1)
                else:
                    this_label = str(site_idx + 1)
                ax[idx, 0].text(site[0] - rad - 1.5, site[1] - rad - 1.5, this_label, color='red')
        if titles is not None:
            ax[idx, 0].set_title(titles[idx])
    fig.tight_layout()
    return fig, ax

def plot_dual_images(imgs, sites=None, box_size=None, box_label=None, titles=None):
    # Similar idea to plot_images, but designed for plotting data from two images. 
    nimgs = len(imgs)
    frame_y = imgs[0].shape[0]
    frame_x = imgs[0].shape[1]
    fig, ax = plt.subplots(nimgs, 1, squeeze=False)
    for idx in range(nimgs):
        this_img = imgs[idx][:,:,0]**2 / np.max(imgs[idx][:,:,0])**2 - imgs[idx][:,:,1] / np.max(imgs[idx][:,:,1])
        ax[idx, 0].imshow(this_img, extent=(-frame_x / 2 + 0.5, frame_x /2 + 0.5, frame_y /2 - 0.5, -frame_y/2 - 0.5), cmap="RdBu_r")
        if sites is not None:
            for i in range(2):
                if utils.is_number(box_size):
                    this_box_size = box_size
                else:
                    this_box_size = box_size[idx][i]
                these_sites = sites[idx][i]
                rad = np.ceil((this_box_size-1)/2) # Following the convention in MATLAB...
                for site_idx, site in enumerate(these_sites):
                    if i == 1:
                        color = 'blue'
                    else:
                        color = 'red'
                    rect = patches.Rectangle((site[0] - rad - 0.5, site[1] - rad - 0.5), 2 * rad + 1, 2 * rad + 1, edgecolor=color, facecolor='none', linewidth=2)
                    ax[idx, 0].add_patch(rect)
                    if box_label is not None:
                        if len(box_label[idx][i]) != 0:
                            this_label = box_label[idx][i][site_idx]
                        else:
                            this_label = str(site_idx + 1)
                    else:
                        this_label = str(site_idx + 1)
                    ax[idx, 0].text(site[0] - rad - 1.5, site[1] - rad - 1.5, this_label, color=color)
        if titles is not None:
            ax[idx, 0].set_title(titles[idx])
    fig.tight_layout()
    return fig, ax
    

def plot_histograms(sigs, plot_titles=None):
    # Assume 1st dimension of sigs will be plotted in each column, and that the 2nd dimension of sigs will be plotted as each row.
    # The third dimension will be the axis used to determine the number of counts
    if isinstance(sigs, list):
        num_imgs = len(sigs)
        num_sites = sigs[0].shape[1]
    else:
        num_imgs = sigs.shape[0] # We use these names only to reflect the relatively simple case of nimgs x nsites x nseqs as the meaning of each axis
        num_sites = sigs.shape[1]
    fig, ax = plt.subplots(num_sites, num_imgs, squeeze=False)
    for col_num in range(num_imgs):
        for row_num in range(num_sites):
            if isinstance(sigs, list):
                ax[row_num, col_num].hist(sigs[col_num][0, row_num, :], bins=40)
            else:
                ax[row_num, col_num].hist(sigs[col_num, row_num, :], bins=40)
            if plot_titles is not None:
                ax[row_num, col_num].set_title(plot_titles[col_num][row_num])
            if row_num == (num_sites - 1):
                ax[row_num, col_num].set_xlabel('Electron counts')
    fig.tight_layout()
    return fig, ax

def plot_survival(x_lists, surv, surv_errs, plot_titles=None, legend_names=None, xlabels=None):
    # Assume 1st dimension is number of columns, 2nd dimension is the number of curves in each plot and 3rd dimension is the independent variable
    num_imgs = surv.shape[0]
    num_sites = surv.shape[1] # Again, names here illustrate a simple use case
    num_params = surv.shape[2]
    if not isinstance(x_lists, list):
        x_lists = [x_lists for i in range(num_imgs)]
    colors = utils.get_colors(num_sites)
    fig, ax = plt.subplots(1, num_imgs, squeeze=False)
    for img_num in range(num_imgs):
        for site_num in range(num_sites):
            if legend_names is None:
                this_label = 'site: ' + str(site_num + 1)
            else:
                this_label = legend_names[img_num, site_num]
            print(x_lists[img_num])
            print(surv[img_num, site_num,:])
            ax[0,img_num].errorbar(x_lists[img_num], surv[img_num, site_num,:], surv_errs[img_num, site_num, :], label=this_label, color=colors[site_num])
        if plot_titles is not None:
            ax[0,img_num].set_title(plot_titles[img_num])
        if xlabels is not None:
            ax[0,img_num].set_xlabel(xlabels[img_num])
        ax[0,img_num].legend()
    fig.tight_layout()
    return fig, ax