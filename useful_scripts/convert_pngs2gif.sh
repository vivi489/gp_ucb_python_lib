#!/bin/bash
#
# Example:
# ./useful_scripts/convert ./samples/gaussian/output_gaussian/res*.png ./samples/gaussian/output_gaussian/anim.gif

input_wildcarded_image_name=$1
output_gif_name=$2

convert -layers optimize -loop 0 -delay 40 $input_wildcarded_image_name $output_gif_name
