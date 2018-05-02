from distutils.core import setup
from setuptools import find_packages

setup(
    name='SafeImage',
    version='0.1',
    packages=find_packages(),
    package_data={'':['*.npy']},
    include_package_data=True,
    url='https://github.com/arquivo/SafeImage',
    license='',
    author='dbicho',
    author_email='daniel.bicho@fccn.pt',
    description='REST Web Service to classify an image if it has adult free content or not',
    #install_requires=['tensorflow', 'Pillow', 'Flask', 'Flask_Autodoc', 'Flask_RESTful', 'numpy']
)
