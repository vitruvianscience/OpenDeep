from __future__ import print_function

from setuptools import setup, find_packages
from setuptools.command.install import install
import io
import os

here = os.path.abspath(os.path.dirname(__file__))

# Grab the appropriate version number from opendeep/version.py so we only have to keep track of it in one place!
exec(compile(open('opendeep/version.py').read(), 'opendeep/version.py', 'exec'))
# now we have the appropriate version in __version__
version = __version__  # it is there, trust me :) IDE's won't recognize that exec does anything.

def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    mode = kwargs.get('mode', 'rb')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        # hotfix for ValueError: binary mode doesn't take an encoding argument.
        if 'b' not in mode:
            with io.open(filename, mode=mode, encoding=encoding) as f:
                buf.append(f.read())
        else:
            with io.open(filename, mode=mode) as f:
                buf.append(f.read())
    return sep.join(buf)

# Inform user of setup.py develop preference
class opendeep_install(install):
    def run(self):
        print("OpenDeep is in alpha and undergoing heavy development. "
              "We recommend using 'python setup.py develop' rather than 'python setup.py install'.")
        mode = None
        while mode not in ['', 'install', 'develop', 'cancel']:
            if mode is not None:
                print("Please try again")
            mode = input("Installation mode: [develop]/install/cancel: ")
        if mode in ['', 'develop']:
            self.distribution.run_command('develop')
        if mode == 'install':
            return install.run(self)

setup(
    name='opendeep',
    version=version,
    description='A modular and extensible deep learning library built on Theano.',
    long_description=read('README.rst'),
    keywords='opendeep theano modular deep learning neural net ai',

    url='https://github.com/vitruvianscience/opendeep/',

    author='Vitruvian Science',
    author_email='opendeep-dev@googlegroups.com',

    license='Apache2',

    classifiers=[
        'Programming Language :: Python',
        'Natural Language :: English',
        'License :: OSI Approved :: Apache Software License',
        # How mature is this project? Common values are
        # 3 - Alpha
        # 4 - Beta
        # 5 - Production/Stable
        'Development Status :: 3 - Alpha',
        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries :: Application Frameworks'
    ],

    dependency_links=['git+http://github.com/Theano/Theano.git#egg=Theano'],
    install_requires=["numpy", "theano", "pillow"],

    packages=find_packages(),
    # If there are data files included in your packages that need to be
    # installed, specify them here. If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    package_data={},
)