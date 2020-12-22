"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt
import numpy as np
from random import randint
from argparse import ArgumentParser
from inference import Network


# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST,MQTT_PORT,MQTT_KEEPALIVE_INTERVAL)
    return client

def draw_boxes(frame,result,args,width,height,count,dis):
    '''draw bouding box onto the frame'''
    (xmin,ymin) = None
    (xmax,ymax) = None
    thickness = 5
    color = (0, 0, 255)

    for box in result[0][0]: # Output shape is 1x1x100x7
        conf = box[2]
        if conf >= 0.5:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax),color, thickness)
            current_count = current_count + 1

            """to print the current count"""
            
            frame_x = frame.shape[1]/2
            frame_y = frame.shape[0]/2
            mid_x = (xmax + xmin)/2
            mid_y = (ymax + ymin)/2
            
            dis = math.sqrt(math.pow(mid_x - frame_x, 2) + math.pow(mid_y - frame_y,2)
            count = 0
    
        if current_count <1:
            count += 1

        if dis >0 and count < 10:
            current_count =1
            count += 1
            if count>100:
                count = 0
                
    return frame, current_count, dis, count


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()

    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    ###n, c, h, w = infer_network.load(model, device, 1, 1, request_id, cpu_extension)[1]
    infer_network.load_model(args.model, args.device, args.cpu_extension)
    network_input_shape = infer_network.get_input_shape()

    ### TODO: Handle the input stream ###
    single_image_mode = False

    if args.input =='CAM':
        args.input = 0
    
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp'):
        single_image_mode = True
        args.input = video_file

    else:
        args.input = video_file
        assert os.path.isfile(video_file)

    ##get and open video capture
    captured = cv2.VideoCapture(args.input)
    captured.open(args.input)
    
    ##get the actual shape of input
    width = int(captured.get(3))
    height = int(captured.get(4))
    input_shape = network_input_shape['image_tensor']
    
    total_count = 0
    duration = 0
    
    ### TODO: Loop until stream is over ###
    while captured.isOpened():

        ### TODO: Read from the video capture ###
        flag, frame =  captured.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        ### TODO: Pre-process the image as needed ###
        image = cv2.resize(frame,(w, h))
        image = image.transpose((2,0,1))
        image = image.reshape(1, *image.shape)

        ### TODO: Start asynchronous inference for specified request ###
        network_input = {'image_tensor':image, 'image_info':image.shape[1:]}
        infer_start = time.time()
        infer_network.exec_net(request_id, image)
        
        ### TODO: Wait for the result ###
        if infer_network.wait() == 0 :
            person_count = 0
            det_time = time.time() - infer_start

            ### TODO: Get the results of the inference request ###
            result = infer_network.get_output(request_id)
            
            ### TODO: Extract any desired stats from the results ###
            draw_bounding_box = draw_boxes(frame, result, prob_threshold, width, height)
            infer_time_message = "Inference time: {:.3f}ms".format(det_time * 1000)
            draw_bounding_box = cv2.putText(draw_bounding_box, infer_time_message,(15,15), cv2.FONT_HERSHEY_COMPLEX,0.45, (255, 0, 0), 1)
            
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###

            if current_count > last_count:
                start_time =  time.time()
                total_count = total_count + current_count - last_count
                client.publish("person", json.dumps({"total" : total_count}))

            if current_count < current_count:
                duration = int(time.time() - start_time)
                client.publish("person/duration", json.dumps({"duration": duration}))
        
        ### TODO: Send the frame to the FFMPEG server ###
        draw_bounding_box = cv2.resize(draw_bounding_box,(width,height))
        sys.stdout.buffer.write(draw_bounding_box)
        sys.stdout.flush()
        
        ### TODO: Write an output image if `single_image_mode` ###
        if single_image_mode:
            cv2.imwrite('output_image.jpg', frame)

    ##release all captured frames and destroy opencvwindows
    captured.release()
    cv2.destroyAllWindows()
    client.disconnect()
    infer_network_clean()

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
