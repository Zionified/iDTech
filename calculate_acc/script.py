import argparse
import sys
import os
import re

from jetson_inference import imageNet
from imagenet import process_images
from jetson_utils import videoSource, videoOutput, Log

# parse the command line
parser = argparse.ArgumentParser(description="Classify a live camera stream using an image recognition DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=imageNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="", nargs='?', help="path to data folder")
parser.add_argument("output", type=str, default="", nargs='?', help="path to result folder")
parser.add_argument("--network", type=str, default="googlenet", help="pre-trained model to load (see below for options)")
parser.add_argument("--topK", type=int, default=1, help="show the topK number of class predictions (default: 1)")

try:
    args = parser.parse_known_args()[0]
except:
    print("")
    parser.print_help()
    sys.exit(0)

# val->type1, type2, type3->image1, image2

images = os.listdir(args.input)
total = len(images)
error = 0

def check_output(category, labels):
    for label in labels:
        if category in label:
            return True
    return False

for image in images:
    image_path = os.path.join(args.input, image)
    output_path = os.path.join(args.output, "test_{}".format(image))
    labels = process_images(image_path, output_path)

    if not check_output("cat", labels):
        error += 1

print("Accuracy: ", (total-error)/total)
    
