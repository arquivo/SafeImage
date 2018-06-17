from images_classifiers.cf.classifiers import CaffeNsfwSqueezeClassifier


def test_classify():
    model = CaffeNsfwSqueezeClassifier(batch_size=1)

    with open('image_test_1.jpg', mode='rb') as input_file:
        result = model.classify(input_file.read())

    assert result[0] == 0.0


def test_batch_classification():
    model = CaffeNsfwSqueezeClassifier(batch_size=2)

    with open('image_test_1.jpg', mode='rb') as input_file:
        image_bytes = input_file.read()
        image_data = [image_bytes, image_bytes]
        result = model.classify_batch(image_data)

        assert result[0] == 0.0
        assert result[1] == 0.0
