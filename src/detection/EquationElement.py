import numpy as np
import skimage.measure

class EquationElement:
    """
    Define an equation element of the video (digit or operator)
    """
    def __init__(self, bbox, image, mask):
        # object bbox within video frame
        self.x0, self.y0 = bbox[0], bbox[1]
        self.x1, self.y1 = bbox[2], bbox[3]

        # the image array of the element : HxWx3
        self.image = image
        self.mask = mask

        # the character represented by the image : + - * ÷ = or 0 1 2 3 4 5 6 7 8
        self.type = ''
        self.value = ''

    def get_Fourier_descr(self, N):
        """
        Return the N first fourier descriptor of the equation element mask. The
        fourier descriptors are translation, rotation and scale invariant.
        """
        # get contour
        contour = self._get_linked_contour(self.mask)
        # build complex signal
        signal = contour[:,1] + 1j * contour[:,0]
        # apply DFT
        f_descriptors = np.fft.fft(signal)
        # Apply invariance
        f_descriptors = f_descriptors[1:] # remove 1st descriptor
        f_descriptors = np.abs(f_descriptors) # keep only the amplitude of descriptors
        f_descriptors = f_descriptors[1:] / f_descriptors[0] # devide all descriptors by the value of the first one
        # keep N features
        feat = f_descriptors[:N]

        return feat

    def _get_linked_contour(self, im):
        """
        Assemble the multiple contours of a mask into a single one.
        The contours are linked by their extremities.
        """
        contour = skimage.measure.find_contours(im, level=False, fully_connected='high')
        contour_all = [contour[0]]
        # link the contours
        for contour_i, contour_t in zip(contour[:-1], contour[1:]):
            link = self._get_contour_link(contour_i[0], contour_t[0])
            contour_all += [link, contour_t]
        return np.concatenate(contour_all, axis=0)

    def _get_contour_link(self, c_i, c_t):
        """
        Compute the link (i.e. sequence of coordinates) between
        two contours (source and target).
        """
        link = []
        coord = c_i.copy()
        while np.any(coord != c_t):
            # make a vertical step toward the target
            if coord[0] < c_t[0]:
                coord[0] += 1
            elif coord[0] > c_t[0]:
                coord[0] -= 1
            # make a horizontal step toward the target
            if coord[1] < c_t[1]:
                coord[1] += 1
            elif coord[1] > c_t[1]:
                coord[1] -= 1

            link.append(coord.copy())
        return np.array(link[:-1])

    def has_overlap(self, bbox, frac=0.9):
        """
        Check whether the passed bbox (x0, y0, x1, y1) overlap with the self by
        at least frac%.
        """
        raise(NotImplementedError)
