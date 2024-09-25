import os

import math
import numpy as np

import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from moviepy.editor import ImageSequenceClip


def draw_pick_and_place(image, start, end, color=(143, 201, 58)):
    ## adjust thickness regarding to the image size
    thickness = max(1, int(image.shape[0] / 100))
    # print('start', start, 'end', end)
    # print('image shape', image.shape)
    
    image = cv2.arrowedLine(
        cv2.UMat(image), 
        start, 
        end,
        color, 
        thickness)
    
    image = cv2.circle(
        cv2.UMat(image), 
        start, 
        thickness*2,
        color, 
        thickness)

    return image #.get().astype(np.int8).clip(0, 255)

def plot_pick_and_place_trajectory(obs, acts, 
    info=None,
    save_png=True, save_path='', 
    action_color= [(143, 201, 58), (0, 114, 187)], 
    title='trajectory', show=False, col = 10):

    row = math.ceil(len(obs)/col)
    T, H, W, C = obs.shape
    print('obs shape', obs.shape)
    fig = plt.figure(figsize=(5*col, 5*row))
    outer = fig.add_gridspec(ncols=1, nrows=1)
    inner = gridspec.GridSpecFromSubplotSpec(row, col, # TODO: magic number
                subplot_spec=outer[0], wspace=0, hspace=0)
    
    print('acts', acts)
    
    act_len = acts.shape[0]
    acts =  acts.reshape(act_len, -1, 4)
    pick_num = acts.shape[1]

    pixel_actions = ((acts + 1.0)/2  * np.asarray([H, W, H, W]).reshape(1, 1, 4)).astype(int)
    
    # if acts1 is not None:
    #     pixel_actions_1 = ((acts1 + 1)/2  * np.asarray([H, W, H, W])).astype(int)
    # if acts2 is not None:
    #     pixel_actions_2 = ((acts2 + 1)/2  * np.array([H,W, H, W])).astype(int)

    thickness = 2

    for i in range(len(obs)):

        ####### 
        ax0 = plt.Subplot(fig, inner[i])
        
        image = obs[i]

        if C >= 3:
            image = image[:, :, :3].astype(int).clip(0, 255)
            if i < len(obs) - 1:
                if acts is not None:
                    print('image shape', image.shape)
                    image = draw_pick_and_place(
                        image[:, :, :3],
                        tuple(pixel_actions[i][0][:2]), 
                        tuple(pixel_actions[i][0][2:]),
                        action_color[0],
                    ).get().clip(0, 255).astype(int)
                    if pick_num == 2:
                        image = cv2.arrowedLine(
                            cv2.UMat(image[:, :, :3].astype(int)), 
                            tuple(pixel_actions[i][1][:2]), 
                            tuple(pixel_actions[i][1][2:]),
                            action_color[1], 
                            thickness)
                        image = image.get().astype(int).clip(0, 255)
        
        ax0.axis('on')
        ax0.imshow(image)
        ax0.set_xticks([])
       
        # write the info as text on the corresponding image
        if info is not None:
            ax0.text(0, 0, info[i], fontsize=12, color='blue')


        ax0.set_yticks([])
        fig.add_subplot(ax0)
    
    if show:
        plt.show()

    if save_png:
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        plt.savefig(os.path.join(save_path, '{}.png'.format(title)), bbox_inches='tight')
    
    plt.close()

## Purely polit image trajectory no action involved
def plot_image_trajectory(obs, 
    save_png=True, save_path='.',
    title='trajectory', show=False, col=10,
    row_lables=None):

    #col = col
    row = math.ceil(len(obs)/col)
    #T, H, W, C = obs.shape
    fig = plt.figure(figsize=(5*col, 5*row))
    outer = fig.add_gridspec(ncols=1, nrows=1)
    inner = gridspec.GridSpecFromSubplotSpec(row, col, # TODO: magic number
                subplot_spec=outer[0], wspace=0, hspace=0)
    
    for i in range(len(obs)):

        ####### 
        ax0 = plt.Subplot(fig, inner[i])
        
        image = obs[i]
        H, W, C = image.shape

        if C >= 3:
            image = image[:, :, :3].astype(int).clip(0, 255)
        
        ax0.axis('on')
        ax0.imshow(image)
        ax0.set_xticks([])
        ax0.set_yticks([])
        fig.add_subplot(ax0)

    # if row_lables is not None:
    #     for i in range(len(row_lables)):
    #         fig.text(0.14, 0.87 - i*0.128, row_lables[i], 
    #                  ha='left',
    #                  va='center', 
    #                  fontsize=25, 
    #                  color='red', fontweight='bold')
    
    if show:
        plt.show()

    if save_png:
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        plt.savefig(os.path.join(save_path, '{}.png'.format(title)), bbox_inches='tight')
    
    plt.close()


### frames: S * H * W * 3 in RGB numpy
def save_video(frames, path='', title='default'):
    frames = frames.clip(0, 255).astype(np.uint8)
    bgr_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]

    if not os.path.exists(path):
        os.makedirs(path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    _, H, W, _ = frames.shape
    writter = cv2.VideoWriter(os.path.join(path, '%s.mp4' % title), fourcc, 30, (W, H))
    for frame in bgr_frames:
        writter.write(frame)
    writter.release()

def show_image(img, window_name=''):
    # Input image has to be displayable
    # The input type is either np int8 [0, 255] or np float [0, 1.0]
    # H*W*3 or H*W
    cv2.imshow(window_name, img)
    cv2.waitKey(1)



def save_numpy_as_gif(array, filename, fps=100, scale=1.0):
    # from https://github.com/Xingyu-Lin/softgym/blob/master/softgym/utils/visualization.py
    """Creates a gif given a stack of images using moviepy
    Notes
    -----
    works with current Github version of moviepy (not the pip version)
    https://github.com/Zulko/moviepy/commit/d4c9c37bc88261d8ed8b5d9b7c317d13b2cdf62e
    Usage
    -----
    >>> X = randn(100, 64, 64)
    >>> gif('test.gif', X)
    Parameters
    ----------
    filename : string
        The filename of the gif to write to
    array : array_like
        A numpy array that contains a sequence of images
    fps : int
        frames per second (default: 10)
    scale : float
        how much to rescale each image by (default: 1.0)
    """

    # ensure that the file has the .gif extension
    fname, _ = os.path.splitext(filename)
    filename = fname + '.gif'

    # copy into the color dimension if the images are black and white
    if array.ndim == 3:
        array = array[..., np.newaxis] * np.ones(3)

    # make the moviepy clip
    clip = ImageSequenceClip(list(array), fps=fps).resize(scale)
    clip.write_gif(filename, fps=fps)
    return clip
