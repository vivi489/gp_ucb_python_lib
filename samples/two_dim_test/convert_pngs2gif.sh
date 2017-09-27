#!/bin/bash
#
# Example:
# ./useful_scripts/convert ./samples/gaussian/output_gaussian/res*.png ./samples/gaussian/output_gaussian/anim.gif

input_wildcarded_image_name=./output/res*.png
output_gif_name=./output/$1
#./output/animated.gif

convert -layers optimize -loop 1 -delay 20 -dispose previous $input_wildcarded_image_name $output_gif_name

