import os
import sys

if sys.version_info < (3, 6):
    sys.exit('ERROR: It requires Python 3.6+')

from setuptools import setup, find_packages

setup(
    name='src',
    description="B",
    url='xxx',
    project_urls={
        'Documentation': 'xx',
    },
    long_description=open(os.path.join(os.path.dirname(__file__), 'README.md'),
                          encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    include_package_data=True,
    setup_requires=[
        'setuptools_git',
        'setuptools_scm',
    ],
    python_requires='>=3.6',
    classifiers=[
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3 :: Only',
    ],
)
