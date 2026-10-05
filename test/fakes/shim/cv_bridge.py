"""Test-only stand-in for cv_bridge.CvBridge (no py3.11 build in this env). Handles the
two calls the bridge / client make, for 8-bit rgb images."""
import numpy as np


class CvBridge:
    def cv2_to_imgmsg(self, img, encoding="rgb8"):
        from sensor_msgs.msg import Image
        img = np.ascontiguousarray(img)
        m = Image()
        m.height, m.width = int(img.shape[0]), int(img.shape[1])
        m.encoding = encoding
        m.is_bigendian = 0
        m.step = int(img.shape[1] * img.shape[2])
        m.data = img.tobytes()
        return m

    def imgmsg_to_cv2(self, msg, desired_encoding="rgb8"):
        return np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.width, -1)
