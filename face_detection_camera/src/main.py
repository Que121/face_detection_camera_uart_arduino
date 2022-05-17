import sys
sys.path.append("../../../../common")
sys.path.append("../")
project_path = sys.path[0] + "/../"
sys.path.append(project_path)
import datetime

from cameracapture import CameraCapture
import presenteragent.presenter_channel as presenter_channel
from acllite_model import AclLiteModel
from acllite_resource import AclLiteResource
from vgg_ssd import VggSsd
from periphery import Serial

# 模型输入路径
MODEL_PATH = project_path + "/model/face_detection.om"
MODEL_WIDTH = 304
MODEL_HEIGHT = 300

# presenter配置文件
FACE_DETEC_CONF= project_path + "/scripts/face_detection.conf"
CAMERA_FRAME_WIDTH = 1280
CAMERA_FRAME_HEIGHT = 720

SCORE = 2

def main():
    """main"""
    # 打开 /dev/ttyAMA1 串口波特率为115200
    ser = Serial("/dev/ttyAMA1", 115200)
    print("uart connection test")
    print("Write to UART")

    # ser.write(b"Hello from Atlas 200 DK\n")
        
    # 初始化ACL 具有多个类别的一组规则执行 N 元组搜索
    acl_resource = AclLiteResource()
    acl_resource.init()
   
    # 创建一个检测网络实例，目前使用vgg_ssd网络。
    # 当检测网络被替换时，在此实例化一个新的网络
    detect = VggSsd(acl_resource, MODEL_WIDTH, MODEL_HEIGHT)
    
    # 加载离线模型
    model = AclLiteModel(MODEL_PATH)

    # 根据配置连接到presenter服务器。 
    # 并在连接失败时结束应用程序的执行
    chan = presenter_channel.open_channel(FACE_DETEC_CONF)
    
    if chan is None:
        print("Open presenter channel failed")
        return
    
    # 打开开发板上的CARAMER0摄像头
    cap = CameraCapture(0)
    while True:
       
        # 从相机获取图像
        image = cap.read()
        if image is None:
            print("Get memory from camera failed")
            break
        
        # 检测网络将图像处理成模型输入数据
        model_input = detect.pre_process(image)
        if model_input is None:
            print("Pre process image failed")
            break
       
        # 发送数据到离线模型推理
        result = model.execute(model_input)
       
        # 检测网络分析推理输出
        jpeg_image, detection_list = detect.post_process(result, image)
        
        # 是否有图像
        if jpeg_image is None:
            print("The jpeg image for present is None")
            break
        
        # box_num = int(result[0][0, 0])
    
        # box_info = result[1][0]
        # for i in range(box_num):
        #     # 识别到的物体的置信度
        #     score = box_info[i, SCORE]

                 
            
        chan.send_detection_data(CAMERA_FRAME_WIDTH, CAMERA_FRAME_HEIGHT, 
                                 jpeg_image, detection_list)


if __name__ == '__main__':
    main()
