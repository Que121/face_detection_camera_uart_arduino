"""vgg_ssd"""
import numpy as np
import sys
sys.path.append("../common")
sys.path.append("../")
from acllite_imageproc import AclLiteImageProc
from presenteragent import presenter_datatype
from periphery import Serial

LABEL = 1
SCORE = 2
TOP_LEFT_X = 3
TOP_LEFT_Y = 4
BOTTOM_RIGHT_X = 5
BOTTOM_RIGHT_Y = 6
ser = Serial("/dev/ttyAMA1", 115200)

class VggSsd(object):
    """vggssd"""
    def __init__(self, acl_resource, model_width, model_height):
        self._acl_resource = acl_resource
        self._model_width = model_width
        self._model_height = model_height
        self._dvpp = AclLiteImageProc(acl_resource)

    def __del__(self):
        if self._dvpp:
            del self._dvpp
        print("Release yolov3 resource finished")
        print("释放yolov3资源完成")

    def pre_process(self, image):
        """Use dvpp to scale the image to the required size of the model"""
        """使用dvpp将图像缩放到模型所需的尺寸。"""
        resized_image = self._dvpp.resize(image, self._model_width,
                                          self._model_height)
        if resized_image is None:
            print("Resize image failed")
            return None
        # 输出经过缩放的图像和图像信息作为推理输入数据
        return [resized_image,]

    def post_process(self, infer_output, origin_img):
        """Analyze inference output data"""
        """分析推理输出数据"""
        detection_result_list = self._analyze_inference_output(infer_output, 
                                                               origin_img)
        # 将yuv图像转换为jpeg图像
        jpeg_image = self._dvpp.jpege(origin_img)
        return jpeg_image, detection_result_list

    def _analyze_inference_output(self, infer_output, origin_img):
        # vgg ssd有两个输出，第一个输出 
        # infer_output[0]是检测到的物体的数量，形状是（1,8）。
        box_num = int(infer_output[0][0, 0])
        #第二个输出infer_output[1]是检测到的物体信息，其形状为（1, 200, 8）。
        box_info = infer_output[1][0]  
        detection_result_list = []
        
        for i in range(box_num):
            # 识别到的物体的置信度
            score = box_info[i, SCORE]
            if score < 0.9:
                break 
            if score > 0.90:
                ser.write(b"1\n")
                # 最多读取32个字节，延时2秒
                readdata = ser.read(32, 2).decode('gbk')
                print(f'Received reply: {readdata}')
            detection_item = presenter_datatype.ObjectDetectionResult()            
            detection_item.confidence = score

            # 人脸位置框架坐标，归一化坐标。
            # 需要乘以图片的宽度和高度来转换为图片上的坐标
            detection_item.box.lt.x = int(box_info[i, TOP_LEFT_X] * origin_img.width)
            detection_item.box.lt.y = int(box_info[i, TOP_LEFT_Y] * origin_img.height)
            detection_item.box.rb.x = int(box_info[i, BOTTOM_RIGHT_X] * origin_img.width)
            detection_item.box.rb.y = int(box_info[i, BOTTOM_RIGHT_Y] * origin_img.height)
            # 将置信度组织成一个字符串
            detection_item.result_text = str(round(detection_item.confidence * 100, 2)) + "%"
            
            detection_result_list.append(detection_item)
            
        return detection_result_list
