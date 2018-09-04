#!/bin/bash

NM="nm"  # GNU nm (GNU Binutils for Debian) 2.30
OBJCOPY="objcopy"  # GNU objcopy (GNU Binutils for Debian) 2.30

SRC_LIB=tensorflow/contrib/tensorrt/libnvinfer_original.a
DST_LIB=tensorflow/contrib/tensorrt/libnvinfer.a
# cp /<trt-installation-dir>/lib/libnvinfer.a $SRC_LIB
# cp $SRC_LIB $DST_LIB

objcopy_rename_options=$(\
  $NM $SRC_LIB      |  # Dump all symbols.
  grep ' [TVW] '   |  # Keep only global symbols.
  cut -d' ' -f3    |  # Cut out the symbol name.
  egrep '^_Z[A-Z]{1,3}11flatbuffers' |  # Keep only the symbols we care about.
  sort |
  uniq |
  xargs -I SYM echo --redefine-sym SYM=tensorrt_SYM)
$OBJCOPY $objcopy_rename_options $SRC_LIB $DST_LIB

# The following command will fail before patching, and succeed after.
# bazel run --verbose_failures --jobs=32 -c opt --copt=-mavx --output_filter=DONT_MATCH_ANYTHING tensorflow/contrib/tensorrt:base_test
