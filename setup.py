# adapted from https://github.com/kennethreitz/setup.py/blob/master/setup.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pipenv install twine --dev

import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

# Package meta-data.
NAME = 'Amumo'
DESCRIPTION = 'Explore CLIP-alike models in your computational notebook.'
URL = 'https://github.com/ginihumer/Amumo'
EMAIL = 'ginihumer96@yahoo.de'
AUTHOR = 'Gin'
REQUIRES_PYTHON = '>=3.9.7'
VERSION = '0.1.38'

# What packages are required for this module to be executed?
# TODO: update the required packages
REQUIRED = [
    'plotly',
    'ipywidgets==8.0.6',
    'ipykernel==6.23.1',
    'scikit-learn==1.3.0',
    'openTSNE==1.0.0',
    'umap-learn==0.5.3',
    'numpy==1.23.5',
    'torch==2.2.0',
    'pillow==10.2.0',
    'requests',
]

CLIP_REQUIRE = [
    'clip @ git+https://github.com/openai/CLIP.git@a9b1bf5920416aaeaec965c25dd9e8f98c864f16',
]
OPEN_CLIP_REQUIRE = [
    'open-clip-torch==2.20.0',
]
CLOOB_REQUIRE = CLIP_REQUIRE + ['torchvision']
CLOOME_REQUIRE = CLOOB_REQUIRE

BLIP_REQUIRE = [
    'transformers==4.31.0',
    'wrapt',
    'termcolor'
]
IMAGEBIND_REQUIRE = [
    'imagebind @ git+https://github.com/facebookresearch/ImageBind@c6a47d6dc2b53eced51d398c181d57049ca59286',
    'soundfile==0.12.1',
    'torchvision',
    'flatbuffers'
]
DIFFUSION_DB_REQUIRE = [
    'datasets==2.14.6',
]
MSCOCO_REQUIRE = [
    'webdataset==0.2.48',
    'pycocotools==2.0.6',
]

# What packages are optional?
EXTRAS = {
    'clip': CLIP_REQUIRE,
    'open-clip': OPEN_CLIP_REQUIRE,
    'cloob': CLOOB_REQUIRE,
    'cloome': CLOOME_REQUIRE,
    'blip': BLIP_REQUIRE,
    'imagebind': IMAGEBIND_REQUIRE,
    'diffusion-db': DIFFUSION_DB_REQUIRE,
    'mscoco': MSCOCO_REQUIRE,
    # dependencies used in the analysis for the VISxAI article: https://github.com/ginihumer/Amumo/blob/main/notebooks/clip_article.ipynb
    'visxai': CLOOB_REQUIRE + OPEN_CLIP_REQUIRE + DIFFUSION_DB_REQUIRE + MSCOCO_REQUIRE
}

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION


class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')

        sys.exit()


# Where the magic happens:
setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]), # ['amumo', 'amumo.CLOOB_local'],
    # If your package is a single module, use this instead of 'packages':
    py_modules=['amumo'],
    package_data={'amumo': ['CLOOB_local/clip/bpe_simple_vocab_16e6.txt.gz', 'CLOOB_local/cloob_training/pretrained_configs/*.json', 'CLOOB_local/example_configs/*.json', 'CLOOB_local/training/model_configs/*.json']},
    # entry_points={
    #     'console_scripts': ['mycli=mymodule:cli'],
    # },
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license='MIT',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
    # $ setup.py publish support.
    cmdclass={
        'upload': UploadCommand,
    },
)