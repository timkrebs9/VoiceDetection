#!/usr/bin/env python
# -*- coding: utf-8 -*-

__version__ = '1.0.0'

import sys
import warnings

# Throw a deprecation warning if we're on legacy python
if sys.version_info < (3,):
    warnings.warn('You are using Python 2.'
                  'Please note that Python 3 or later is required.',
                  FutureWarning)