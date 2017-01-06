from distutils.core import setup

setup(
    name='SafeImage',
    version='0.1',
    packages=['images_classifiers'],
    url='https://github.com/danielbicho/SafeImage',
    license='',
    author='dbicho',
    author_email='daniel.bicho@fccn.pt',
    description='REST Web Service to classify an image if it has adult free content or not',
    install_requires=['tensorflow', 'Pillow', 'Flask', 'Flask_Autodoc', 'Flask_RESTful' ]
)
