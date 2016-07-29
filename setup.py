#!/usr/bin/env python
# coding: utf-8
from setuptools import setup, find_packages
from Morp import __author__, __version__, __license__
 
setup(
        name             = 'Morp',
        version          = __version__,
        description      = '点予測単語分割'
        license          = __license__,
        author           = __author__,
        author_email     = 'd460655046504@gmail.com',
        url              = 'https://github.com/junion-org/pip_github_test.git',
        keywords         = 'nlp, word_segmentation, japanese',
        packages         = find_packages(),
        install_requires = [],
        )
