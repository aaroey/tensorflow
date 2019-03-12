#!/usr/bin/python3
from __future__ import division, print_function

import tensorflow as tf


def main():
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config_str = config.SerializeToString()
  config_ascii = ', '.join(str(c) for c in config_str)

  print('static const unsigned char session_config[] = {{{}}};'.format(
      config_ascii))


if __name__ == '__main__':
  main()
