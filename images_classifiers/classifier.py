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

    def classify_batch(self, images_data):
        """Classify a batch of images.

        Args:
            images_data: An array of images bytes to classify.
        Returns:
            The classifaction result.
        """
        raise NotImplementedError
