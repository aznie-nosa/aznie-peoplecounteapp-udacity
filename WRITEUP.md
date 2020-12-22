# Project Write-Up

This is a people counter application project from Udacity platform that utilize OpenVINO toolkit to convert the models and doing the inference of the model.
This project was made as one of the requirements in order to obtain Intel(R) Edge AI for IoT Developers Nanodegree program at Udacity
## Explaining Custom Layers
According to the documentation of the Intel® Distribution of OpenVINO™ toolkit custom layers are layers that are not in the list of known layers. OpenVINO have a list of known layers that supported different framework. To see the layers supported by the framework, refer to supported frameworks in the OpenVINO Toolkit official documentation.
If your topology contains any layers that are not in the list of known layers, the Model Optimizer classifies them as custom layers.

Here are the description of the term use in OpenVINO Toolkit distribution that we will see in this project:
	1. Layer — The abstract concept of a math function that is selected for a specific purpose (relu, sigmoid, tanh, convolutional). This is one of a sequential series of building blocks within the neural network.
	2.Kernel — The implementation of a layer function, in this case, the math programmed (in C++ and Python) to perform the layer operation for target hardware (CPU or GPU).
	3.Intermediate Representation (IR) — Neural Network used only by the Inference Engine in OpenVINO abstracting the different frameworks and describing topology, layer parameters and weights. The original format will be a supported framework such as TensorFlow, Caffe, or MXNet.
	4.Model Extension Generator — Generates template source code files for each of the extensions needed by the Model Optimizer and the Inference Engine.
	5.Inference Engine Extension — Device-specific module implementing custom layers (a set of kernels).

##Result summary and Model info

There are 3 models that I downloaded and testing with the application. However,in my investigation of potential models for the people counter app, I found a suitable model that works well for the purpose of our app. Below is the information about the model:
Model :ssd_mobilenet_v2_coco_2018_03_29
Framework : TensorFlow
Size of the model : 180MB

For this people counter app, I choose SSD MobileNet V2 COCO model and here is the steps to download the model:
First,download the model from TensorFlow by executing this: 
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz

The model downloaded is in .tar file, need to use tar -xvf command to unpack the model.

tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz

You will see ssd_mobilenet_v2_coco_2018_03_29 folder and the content of the folder is : 
- saved_model folder 
- checkpoint
- frozen_inference_graph.mapping
- frozen_inference_graph.pb
- model.ckpt.data-00000-of-00001
- model.ckpt.index
- model.ckpt.meta
- pipeline.config

Then, convert the model into the Intermediate Representation(IR) format by using the following command:

python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json

Later, after the conversion is successful, you will see  frozen_inference_graph.bin and frozen_inference_graph.xml in the folder.
##How to run the project

Using a video file, please execute the following command:
- python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm 

To use camera stream, run the following command :
- python main.py -i CAM -m your-model.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were:

I compare the performance of this three model and pick the best from the options.

1. ssd_mobilenet_v2_coco_2018_03_29 model:
- model accuracy : pre and post-conversion
- size of the model :180MB

2. squeezenet_v1.1.caffemodel:
- model accuracy : pre and post-conversion
- size of the model: 4.885MB

3.  human-pose-estimation-0001 model:
- model accuracy : pre and post-conversion
- size of the model : 16.152MB

The way that I compare the models is by running the application. The app is running using the 3 models I and compare the accuracy and the inference time of the result. Between this 3 models, the SSD MobileNet V2 COCO have the biggest size but the perfomance is better than other 2 models.
The inference time of the model pre- and post-conversion was 70ms. I've tested noth pretrained model and the converted model and the result is the pretrained model from openmodel zoo have a lesser inference time compare to the converted model. Same as the accuracy of the detection of the pretrained model.

##Downloading the model

This is the process of generating Intermediate Representation (IR) models from OpenVINO Toolkit,
1. Firstly, download the OpenVINO Toolkit.
2. Open Command Prompt
3. Initialize the OpenVINO environment : cd <INSTALL_DIR>\Intel\openvino\bin\source setupvar.sh
4. Go to model_downloader directory : cd <INSTALL_DIR>\Intel\openvino\deployment_tools\tools\model_downloader\downloader.py --name ssd_mobilenet_v2_coco
5. Find the models downloaded in <INSTALL_DIR>\Intel\openvino\deployment_tools\tools\model_downloader\downloader\public directory

##Converting the Model to Intermediate Representation (IR)
1. Open Command Prompt
2. Initialize the OpenVINO environment : cd <INSTALL_DIR>\Intel\openvino\bin\source setupvar.sh
3. Go to Model Optimizer directory : cd <INSTALL_DIR>\Intel\openvino\deployment_tools directory.
4. Execute python mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config <INSTALL_DIR>\openvino\deployment_tools\model_optimizer\extensions\front\tfssd_v2_support.json

## Assess Model Use Cases
A people counter is an application to measure the number of people traversing a certain place, entrance or areas.This application can be implement in retails stores, shopping malls, public transport and many more. The application can be use to monitor a high-traffic areas to measure the number of vistor in a stores so that we can set a limit of customer in a certain time.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. 

- Lighting : The ligthing of the camera/image is important as it can affect the perfomance of the image processing. A stable detection requires projection of a clear image. The lighting selection plays an important role in determining the performance of image processing–based inspection. 
			If the application in running in an outdoor area, the lighthing may affect by the sunlight or a at night it might be affacted by street light or any lighting sources. So to have a better perfromnace, the ligthing condition must be consider for the app to be more accurate anf detection is good.
			
- Model Accuracy : The model accuracy is proportional to the model perfromance. The better the performance of the model the accurate the prediction of the application.
				   The deployed edge model must have high accuracy since the models result is requires for a real time usage. Any incorrect prediction will give a poor result in the performance. The model accuracy also depend also effected by the inference time.So it is important to use the model that give high performance of the application.
				   
- Camera focal length/image size : The focal length is important as it dictates how much of the scene the camera will be able to capture. The image size is also important to have a clear image for the application to do the prediction.
								   Focal length is the distance between the point of convergence of your lens and the sensor or film recording the image. The focal length of your film or digital camera lens dictates how much of the scene your camera will be able to capture.
								   The smaller numbers have a wider angle of view and show more of the scene, while larger numbers have a narrower angle of view and show less.
								   The focal lentgh affect an image by field of view, depth of field,perspective and also image shake. Field of view is determined the scene that can captured by the camera. The depth of field means that the camera can fokus on a small object at specific distances. Image shake is the blurriness and reduction in image quality that occurs from the vibration.
								   To run the application with your camera, it is important to make sure consider this circumstances to get a good result.
## Model Research
 

In investigating potential people counter models, I tried each of the following three models:

- Model 1: ssd_mobilenet_v2_coco
  - [Model Source]:OpenVINO™ model downloader or wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
  - The model downloaded will be in .tar,need to execute : tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
  - I converted the model to an Intermediate Representation with the following arguments : python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config <INSTALL_DIR>/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
  
- Model 2: squeezenet_v1.1.caffemodel
  - [Model Source]:https://github.com/forresti/SqueezeNet
  - I converted the model to an Intermediate Representation with the following steps.
  - Go to the <INSTALL_DIR>/deployment_tools/model_optimizer directory, run following arguments : python mo.py --input_model squeezenet_v1.1.caffemodel
  - The model is insufficient for the app because: output blob: ["prob"], output shape: (10, 1000, 1, 1) not fit to app's process_output_bb() def; need shape [1x1xNofBoxesx7]
  
- Model 3: human-pose-estimation-0001
  - [Model Source]: OpenVINO™ model downloader 
  - I model is directky downloaded the .xml and .bin file from Model Downloader vis this command : downloader.py --name human-pose-estimation-0001
  - The model was insufficient for the app because frame per second (FPS) is low.
  
*** After the reserch between this 3 models,I have choose to use the SSD MobileNet V2 COCO model for my people counter app.