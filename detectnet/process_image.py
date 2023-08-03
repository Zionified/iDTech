#!/usr/bin/env python3
#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import sys

from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput


def process_image(input, output, network="ssd-mobilenet-v2", threshold=0.5, overlay="box", DEFAULT=False):

    # load the recognition network
    if DEFAULT:
        net = detectNet(network, sys.argv, threshold)
    else:
        net = detectNet(model="models/fox/ssd-mobilenet.onnx", labels="models/fox/labels.txt", 
                input_blob="input_0", output_cvg="scores", output_bbox="boxes", 
                threshold=threshold)
    
    input = videoSource(input, argv=sys.argv)
    output = videoOutput(output, argv=sys.argv)

    if input is None: # timeout
        return  
    
    img = input.Capture()

    detections = net.Detect(img, overlay=overlay)
    
    objects = {}

    for detection in detections:
        label = net.GetClassDesc(detection.ClassID)
        confidence = detection.Confidence 
        objects[label] =  confidence
    
    # render the image
    output.Render(img)

    return objects