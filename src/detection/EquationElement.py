class EquationElement:
    """
    Define an equation element of the video (digit or operator)
    """
    def __init__(self, bbox, image):
        # object bbox within video frame
        self.x0, self.y0 = bbox[0], bbox[1]
        self.x1, self.y1 = bbox[2], bbox[3]

        # the character represented by the image : + - * ÷ = or 0 1 2 3 4 5 6 7 8
        self.value = ''

        # the image array of the element : HxWx3
        self.image = image
        self.mask = mask

    def get_Fourier_descr(self, N=None):
        """
        Return the N first fourier descriptor of the equation element mask. The
        fourier descriptors are translation, rotation and scale invariant.
        """
        raise(NotImplementedError)

    def has_overlap(self, bbox, frac=0.9):
        """
        Check whether the passed bbox (x0, y0, x1, y1) overlap with the self by
        at least frac%.
        """
        raise(NotImplementedError)
