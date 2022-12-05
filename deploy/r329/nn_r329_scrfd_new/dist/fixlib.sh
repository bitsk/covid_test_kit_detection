#!/bin/sh

curr_dir=$(cd `dirname $0`; pwd)

ln -s $curr_dir/lib/libopencv_core.so $curr_dir/lib/libopencv_core.so.4.5
ln -s $curr_dir/lib/libopencv_freetype.so $curr_dir/lib/libopencv_freetype.so.4.5
ln -s $curr_dir/lib/libopencv_highgui.so $curr_dir/lib/libopencv_highgui.so.4.5
ln -s $curr_dir/lib/libopencv_imgcodecs.so $curr_dir/lib/libopencv_imgcodecs.so.4.5
ln -s $curr_dir/lib/libopencv_imgproc.so $curr_dir/lib/libopencv_imgproc.so.4.5
ln -s $curr_dir/lib/libopencv_videoio.so $curr_dir/lib/libopencv_videoio.so.4.5
ln -s $curr_dir/lib/libaipudrv.so $curr_dir/lib/libaipudrv.so.5

