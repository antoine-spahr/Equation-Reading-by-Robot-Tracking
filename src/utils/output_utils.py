import numpy as np
import imageio
import matplotlib.pyplot as plt
import matplotlib

def draw_output_frame(frame, trajectory, equation_str, eq_elem_list=None):
    """
    Draw the frame to be saved by drawing the trajectory and the equation on top
    of the original frame.
    """
    traj = np.array(trajectory)
    # creat figure that matches the input dimension
    xpixels = frame.shape[1]
    ypixels = frame.shape[0] + 48
    dpi = 72
    scalefactor = 1
    xinch = xpixels * scalefactor / dpi
    yinch = ypixels * scalefactor / dpi

    fig, ax = plt.subplots(1, 1, figsize=(xinch,yinch))
    canvas = matplotlib.backends.backend_agg.FigureCanvas(fig)
    fig.set_facecolor('black')
    # plot frame
    ax.imshow(frame, aspect='equal')
    # plot trajectory since begining
    if eq_elem_list:
        o = 3
        for elem in eq_elem_list:
            if elem.type == 'digit':
                color = 'black'
            elif elem.type == 'operator':
                color = 'royalblue'
            elif elem.type == 'arrow':
                color = 'crimson'
            else:
                raise ValueError('Wrong type for element, must be operator, digit or arrow')

            ax.add_patch(matplotlib.patches.Rectangle((elem.x0-o, elem.y0-o),
                                                elem.x1-elem.x0+2*o, elem.y1-elem.y0+2*o,
                                                fill=False, ec=color, lw=2, alpha=0.5))
            ax.annotate(elem.value, (elem.x0-o, elem.y0-o), fontsize=12, fontweight='bold', color='darkgray',
                        ha='left', va='bottom', xytext=(3,4), textcoords='offset points',
                        bbox=dict(fc=color, lw=0, alpha=0.5))

    ax.plot(traj[:,0], traj[:,1], color='crimson', lw=0, marker='o', mew=1, mec='gray', mfc='black', markersize=15, alpha=0.4)
    ax.text(10, ypixels-43, 'Equation : '+equation_str, fontsize=24, fontweight='bold', color='white', ha='left', va='top')

    # plot equation string
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    ax.margins(0)
    canvas.draw()
    # get figure as np.array
    out_frame = np.array(canvas.renderer.buffer_rgba(), dtype=np.uint8)
    plt.close()

    return out_frame

def save_video(filename, frame_list, fps):
    """
    Save the list of frame as a video.
    """
    writer = imageio.get_writer(filename, fps=fps)
    for frame in frame_list:
        writer.append_data(frame)
    writer.close()
