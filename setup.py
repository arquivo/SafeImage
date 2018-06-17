from distutils.core import setup
from setuptools import find_packages

setup(
    name='SafeImage',
    version='1.0',
    packages=find_packages(),
    package_data={'': ['*.npy']},
    include_package_data=True,
    url='https://github.com/arquivo/SafeImage',
    license='',
    author='dbicho',
    author_email='daniel.bicho@fccn.pt',
    description='REST Web Service to classify an image if it have adult free content or not',
    install_requires=[
        'PyYAML',
        'redis',
        'requests',
        'numpy',
        'tensorflow',
        'Pillow',
        'Flask',
        'Flask_Autodoc',
        'Flask_RESTful',
        'uwsgi',
        'scipy',
        'sklearn',
        'scikit-image'
    ],

    entry_points={
        'console_scripts': [
            'safe-image-api=safe_image_api:main',
            'nsfw-resnet-worker=workers.resnet_nsfw_worker:main',
            'nsfw-squeezenet-worker=workers.squeezenet_nsfw_worker:main'
        ],
    },
)
