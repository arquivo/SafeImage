from images_classifiers.cf.classifiers import CaffeImageNetVGG19Classifier


def test_classify():
    model = CaffeImageNetVGG19Classifier(batch_size=1, mode_gpu=False)

    with open('./test_files/gato_1.png', mode='rb') as input_file:
        result = model.classify(input_file.read())

    assert result[0] == 'kuvasz'


def test_batch_classification():
    model = CaffeImageNetVGG19Classifier(batch_size=2, mode_gpu=False)

    with open('./test_files/gato_1.png', mode='rb') as input_file:
        image_bytes = input_file.read()
        image_data = [image_bytes, image_bytes]
        result = model.classify_batch(image_data)

        assert result[0][0] == 'kuvasz'
        assert result[1][1] == 'Great Pyrenees'
