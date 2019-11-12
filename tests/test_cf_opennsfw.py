from images_classifiers.cf.classifiers import CaffeNsfwResnetClassifier


def test_classify():
    model = CaffeNsfwResnetClassifier(batch_size=1, mode_gpu=False)

    with open('test_files/image_test_1.jpg', mode='rb') as input_file:
        result = model.classify(input_file.read())

    assert result == 0.002


def test_batch_classification():
    model = CaffeNsfwResnetClassifier(batch_size=2, mode_gpu=False)

    with open('test_files/image_test_1.jpg', mode='rb') as input_file:
        image_bytes = input_file.read()
        image_data = [image_bytes, image_bytes]
        result = model.classify_batch(image_data)

        assert result[0] == 0.002
        assert result[1] == 0.002
