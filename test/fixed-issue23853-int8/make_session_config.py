#!/usr/bin/env python

from __future__ import division, print_function

import tensorflow as tf


def main():
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  print('static const unsigned char session_config[] = {{{}}};'.format(
      ', '.join(str(c) for c in config.SerializeToString())))


if __name__ == '__main__':
  main()
