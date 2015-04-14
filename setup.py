from __future__ import print_function

from setuptools import setup, find_packages
from setuptools.command.install import install
from os import path

# Inform user of setup.py develop preference
class opendeep_install(install):
    def run(self):
        print("OpenDeep is in alpha and undergoing heavy development. We recommend using 'python setup.py develop' rather than 'python setup.py install'.")
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
    version='0.0.6a',
    description='A modular deep learning library built on Theano.',
    long_description=open('README.rst', 'rb').read().decode('utf8'),
    keywords='opendeep theano modular deep learning neural',

    url='https://github.com/vitruvianscience/opendeep',

    author='Vitruvian Science',
    author_email='opendeep-dev@googlegroups.com',

    license='Apache2',

    classifiers=[
        # How mature is this project? Common values are
        # 3 - Alpha
        # 4 - Beta
        # 5 - Production/Stable
        'Development Status :: 3 - Alpha',
        # Indicate who your project is intended for
        'Intended Audience :: Developers'],

    dependency_links=['git+http://github.com/Theano/Theano.git#egg=Theano'],
    install_requires=['numpy>=1.5', "Theano"],

    packages=find_packages(),
    # If there are data files included in your packages that need to be
    # installed, specify them here. If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    package_data={},
)
