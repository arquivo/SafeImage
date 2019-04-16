import time

from images_classifiers.cf.classifiers import CaffeNsfwResnetClassifier


def test_classify():
    model = CaffeNsfwResnetClassifier(batch_size=1, mode_gpu=True)

    with open('test_files/image_test_1.jpg', mode='rb') as input_file:
        result = model.classify(input_file.read())

    assert result == 0.002


def test_batch_classification():
    model = CaffeNsfwResnetClassifier(batch_size=2, mode_gpu=True)

    with open('test_files/image_test_1.jpg', mode='rb') as input_file:
        image_bytes = input_file.read()
        image_data = [image_bytes, image_bytes]
        result = model.classify_batch(image_data)

        assert result[0] == 0.002
        assert result[1] == 0.002


def test_batch_speed_classifaction():
    model = CaffeNsfwResnetClassifier(batch_size=16, mode_gpu=True)

    with open('test_files/image_test_1.jpg', mode='rb') as input_file:
        image_bytes = input_file.read()
        image_data = []

        # build up list
        for i in range(16):
            image_data.append(image_bytes)

        start = time.time()
        model.classify_batch(image_data)
        end = time.time()

        time_batch_16 = end - start

        model = CaffeNsfwResnetClassifier(batch_size=1, mode_gpu=True)

        start = time.time()
        model.classify(image_bytes)
        end = time.time()

        time_batch_1 = end - start

        assert (16 / time_batch_16) <= time_batch_1
