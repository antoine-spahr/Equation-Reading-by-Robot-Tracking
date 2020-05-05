

class Detector:
    """

    """
    def __init__(self, frame, digit_model_path, operator_model_path, color_model_path):
        self.frame = frame
        self.element_list = None

        # load the models

    def analyse_frame(self):
        """

        """
        raise(NotImplementedError)
        # get mask
        # get element
        # classify element if they are op or digit
        # classify digit
        # classify op

        # return a list of Equation Element

    def get_mask(self):
        """

        """
        raise(NotImplementedError)

    def extarct_equation_element(self):
        """

        """
        raise(NotImplementedError)

    def classify_color(self):
        """
        Kmeans closest center
        --> evaluation
        """
        raise(NotImplementedError)

    def classify_digit(self):
        """
        MLP on fourier descriptors
        --> evaluation
        """
        raise(NotImplementedError)

    def classify_operator(self):
        """
        1-NN on fourier descriptors
        --> evaluation
        """
        raise(NotImplementedError)

    def draw_frame(self):
        """

        """
        raise(NotImplementedError)
