import numpy as np
import imageio
import matplotlib.pyplot as plt
import matplotlib

def draw_output_frame(frame, trajectory, equation_str):
    """
    Draw the frame to be saved by drawing the trajectory and the equation on top
    of the original frame.
    """
    # creat figure that matches the input dimension
    xpixels = frame.shape[1]
    ypixels = frame.shape[0]
    dpi = 72
    scalefactor = 1
    xinch = xpixels * scalefactor / dpi
    yinch = ypixels * scalefactor / dpi

    fig, ax = plt.subplots(1, 1, figsize=(xinch,yinch))
    # plot frame
    ax.imshow(frame, aspect='equal')
    # plot trajectory since begining
    ax.plot(trajectory[:,0], trajectory[:,1], color='crimson', lw=0, marker='o', mew=1, mec='gray', mfc='black', markersize=15, alpha=0.4)
    # plot equation string
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    ax.margins(0)
    fig.canvas.draw()
    # get figure as np.array
    out_frame = np.array(fig.canvas.renderer.buffer_rgba(), dtype=np.uint8)
    plt.close()

    return out_frame

def save_video(filename, frame_list, fps):
    """
    Save the list of frame as a video.
    """
    writer = imageio.get_writer(filename, fps=fps)
    for fram in frame_list:
        writer.append_data(frame)
    writer.close()
