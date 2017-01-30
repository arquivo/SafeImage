class Classifier():
    """Defines the Classificator Interface."""

    def classify(self, image_data):
        """Classify an image.

        Args:
            image_data: An image bytes to classify.
        Returns:
            The classifaction result.
        """
        raise NotImplementedError
