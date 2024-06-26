import os

from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.txt')) as f:
    README = f.read()
with open(os.path.join(here, 'CHANGES.txt')) as f:
    CHANGES = f.read()

requires = [
    'plaster_pastedeploy',
    'pyramid',
    'pyramid_chameleon',
    'pyramid_debugtoolbar',
    'pyramid_celery',
    'cornice',
    'h5py',
    'mmh3',
    'waitress',
    'opencv-python',
    'pycocotools',
    'matplotlib',
    'onnxruntime',
    'onnx',
    'sqlalchemy',
    'redis',
    'torch',
    'torchvision',
    'torchaudio',
    'rdflib',
    'sparqlwrapper',
    'oxrdflib',
#    'rdflib-endpoint[oxigraph,cli]',
    'pyoxigraph'
]

tests_require = [
    'WebTest',
    'pytest',
    'pytest-cov',
]

setup(
    name='shores_server',
    version='0.0',
    description='Shores image processing server',
    long_description=README + '\n\n' + CHANGES,
    classifiers=[
        'Programming Language :: Python',
        'Framework :: Pyramid',
        'Topic :: Internet :: WWW/HTTP',
        'Topic :: Internet :: WWW/HTTP :: WSGI :: Application',
    ],
    author='',
    author_email='',
    url='',
    keywords='web pyramid pylons',
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    zip_safe=False,
    extras_require={
        'testing': tests_require,
    },
    install_requires=requires,
    entry_points={
        'paste.app_factory': [
            'main = shores_server:main',
        ],
    },
)
