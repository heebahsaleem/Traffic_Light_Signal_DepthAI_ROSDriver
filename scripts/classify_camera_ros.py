# import the necessary packages
from src import config
from src import utils
from imutils.video import FPS
import numpy as np
import cv2
import depthai as dai
import rospy # for ROS 
from sensor_msgs.msg import Image # for ROS 
from cv_bridge import CvBridge # for ROS 

############## Ros Node start Heebah ############
#initialize ros node
rospy.init_node('depth_ros_node', anonymous=True)

#create publisher for the camera frames
image_pub = rospy.Publisher('camera_image', Image, queue_size=10)

############## Ros Node Ends Heebah ############


# initialize a depthai camera pipeline
print("[INFO] initializing a depthai camera pipeline...")
pipeline = utils.create_pipeline_camera()

frame = None
fps = FPS().start()

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(
    config.OUTPUT_VIDEOS,
    fourcc,
    20.0,
    config.CAMERA_PREV_DIM
)

# Pipeline defined, now the device is assigned and pipeline is started
with dai.Device(pipeline) as device:

    # Output queues will be used to get the rgb frames and
    # nn data from the outputs defined above
    q_rgb = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
    q_nn = device.getOutputQueue(name="nn", maxSize=1, blocking=False)

    ########### interface between ROS and OpenCV start Heebah #############
    bridge = CvBridge()
    ########### interface between ROS and OpenCV Ends Heebah #############



     # `get_frame()` fetches frame from OAK,
    # resizes and returns the frame to the calling function
    def get_frame():
        in_rgb = q_rgb.get()
        new_frame = np.array(in_rgb.getData())\
            .reshape((3, in_rgb.getHeight(), in_rgb.getWidth()))\
            .transpose(1, 2, 0).astype(np.uint8)
        new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
        return True, np.ascontiguousarray(new_frame)

    result = None

    ################ ROS shutdown per frames start Heebah #################
    while not rospy.is_shutdown():
        read_correctly, frame = get_frame()

        if not read_correctly:
            break
    ################ ROS shutdown per frames Ends Heebah #################



    # while True:
    #     read_correctly, frame = get_frame()

    #     if not read_correctly:
    #         break

        # fetch neural network prediction
        in_nn = q_nn.tryGet()
        # apply softmax on predictions and
        # fetch class label and confidence score
        if in_nn is not None:
            data = utils.softmax(in_nn.getFirstLayerFp16())
            result_conf = np.max(data)
            if result_conf > 0.5:
                result = {
                    "name": config.LABELS[np.argmax(data)],
                    "conf": round(100 * result_conf, 2)
                }
            else:
                result = None
        # if the prediction is available,
        # annotate frame with prediction and display it
        if result is not None:
            cv2.putText(
                frame,
                "{}".format(result["name"]),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                thickness=2,
                lineType=cv2.LINE_AA
            )
            cv2.putText(
                frame,
                "Conf: {}%".format(result["conf"]),
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                thickness=2,
                lineType=cv2.LINE_AA
            )



        ########## publish camera frame to ROS starts Heebah ############
        ros_image = bridge.cv2_to_imgmsg(frame, encoding="rgb8")
        image_pub.publish(ros_image)
        ########## publish camera frame to ROS ends Heebah ############
        

        # update the FPS counter
        fps.update()

        # display the output
        cv2.imshow("rgb", frame)

        # break out of the loop if `q` key is pressed
        if cv2.waitKey(1) == ord('q'):
            break

        # write the annotated frame to the file
        out.write(frame)

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
out.release()
cv2.destroyAllWindows()
