from distutils.core import setup
from setuptools import find_packages

setup(
    name='SafeImage',
    version='1.0.4',
    packages=find_packages(),
    package_data={'': ['*.npy', '*.prototxt', '*.caffemodel', '*.binaryproto']},
    include_package_data=True,
    url='https://github.com/arquivo/SafeImage',
    license='',
    author='dbicho',
    author_email='daniel.bicho@fccn.pt',
    description='SafeImage Arquivo.pt Classification Tools',
    install_requires=[
        'argparse==1.2.1',
        'pytest==2.8.7',
        'PyYAML==5.3',
        'redis==3.3.11',
        'requests==2.22.0',
        'numpy==1.16.5',
        'tensorflow<=2.0.0',
        'Pillow==8.1.1',
        'flask==1.1.1',
        'Flask_Autodoc==0.1.2',
        'Flask_RESTful==0.3.7',
        'uwsgi==2.0.18',
        'scipy==1.2.2',
        'scikit-learn==0.20.0',
        'scikit-image==0.14.5'
    ],

    entry_points={
        'console_scripts': [
            'safe-image-api=safe_image_api:main',
            'nsfw-resnet-worker=workers.resnet_nsfw_worker:main',
            'nsfw-squeezenet-worker=workers.squeezenet_nsfw_worker:main',
            'cli-safeimage-test-tool=tests.cli_models_test:main',
            'cli-safeimage-indexing=indexing.classify_images_index:main'
        ],
    },
)
