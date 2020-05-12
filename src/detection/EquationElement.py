import numpy as np
import skimage.measure
import itertools
from scipy.spatial.distance import cdist

class EquationElement:
    """
    Define an equation element of the video (digit or operator)
    """
    def __init__(self, image, mask, bbox=(0,0,0,0)):
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

    # def _get_linked_contour(self, im):
    #     """
    #     Assemble the multiple contours of a mask into a single one.
    #     The contours are linked by their extremities.
    #     """
    #     contour = skimage.measure.find_contours(im, level=False, fully_connected='high')
    #     contour_all = [contour[0]]
    #     # link the contours
    #     for contour_i, contour_t in zip(contour[:-1], contour[1:]):
    #         link = self._get_contour_link(contour_i[0], contour_t[0])
    #         contour_all += [link, contour_t]
    #     return np.concatenate(contour_all, axis=0)

    def _get_linked_contour(self, m):
        """
        Assemble the multiple contours of a mask into a single one.
        The contours are linked at their closest point.
        """
        # get image contour
        contour = skimage.measure.find_contours(m, level=False, fully_connected='high')
        n_link = len(contour)-1
        if n_link > 0:
            # get the position for linking with minimal distance
            comb = []
            for (i, j) in itertools.combinations(range(len(contour)), 2):
                d_ij = cdist(contour[i], contour[j])
                pos_ij = np.argwhere(d_ij == d_ij.min())[0]
                min_ij = d_ij.min()
                comb.append(np.array([i, j, pos_ij[0], pos_ij[1], min_ij]))
            # Keep only the smallest number of shorter link
            comb = np.stack(comb, axis=0).astype(int)
            idx = np.argpartition(comb[:,4], n_link-1)
            comb = comb[idx[:n_link],:-1]#.astype(int)
            # merge the first two contour
            linked_contour = self._merge_contour(contour[comb[0,0]], contour[comb[0,1]], comb[0,2], comb[0,3])
            cont_included = [comb[0,0], comb[0,1]]
            # merged other contours
            for comb_i in comb[1:,:]:
                # if comb_i has new contour
                if (comb_i[0] in cont_included) and (not comb_i[1] in cont_included):
                    # find position of linked coordinated in the linked contour
                    pos_linked = int(np.argwhere((linked_contour == contour[comb_i[0]][comb_i[2]]).all(axis=1))[0])
                    linked_contour = self._merge_contour(linked_contour, contour[comb_i[1]], pos_linked, comb_i[3])
                    cont_included.append(comb_i[1])
                elif (not comb_i[0] in cont_included) and (comb_i[1] in cont_included):
                    # find position of linked coordinated in the linked contour
                    pos_linked = int(np.argwhere((linked_contour == contour[comb_i[1]][comb_i[3]]).all(axis=1))[0])
                    linked_contour = self._merge_contour(contour[comb_i[0]], linked_contour, comb_i[2], pos_linked)
                    cont_included.append(comb_i[0])
        else:
            linked_contour = contour[0]

        return linked_contour

    def _merge_contour(self, c1, c2, idx_1, idx_2):
        """
        Link the two contour at the passed idx location.
        """
        # get value of connection points
        p_i = c1[idx_1]
        p_t = c2[idx_2]
        # build link
        link = self._get_contour_link(p_i, p_t)
        # rearrange coord list so that the connection points are the extremities
        c1_corr = np.concatenate([c1[idx_1:-1,:], c1[:idx_1+1,:]], axis=0)
        c2_corr = np.concatenate([c2[idx_2:-1,:], c2[:idx_2+1,:]], axis=0)
        # link them together
        merged_c = np.concatenate([c1_corr, link, c2_corr, link[::-1,:], np.expand_dims(c1_corr[0], axis=0)], axis=0)
        return merged_c

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
        at least frac. Assume the bbox t0 ba larger than the equation element.
        """
        is_overlapping = False
        x0, y0, x1, y1 = bbox

        # check if any overlap
        x0_max, x1_min = max(self.x0, x0), min(self.x1, x1)
        y0_max, y1_min = max(self.y0, y0), min(self.y1, y1)

        if (x0_max < x1_min) and (y0_max < y1_min):
            # compute areas
            overlap_area = (x1_min - x0_max) * (y1_min - y0_max)
            bbox_area = (self.x1 - self.x0) * (self.y1 - self.y0)
            # check if overlap is large enough
            if overlap_area / bbox_area > frac:
                is_overlapping = True

        return is_overlapping
