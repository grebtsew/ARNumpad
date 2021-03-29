import threading
import cv2 


class camera_handler(threading.Thread):
  def __init__(self, camera, shared):
    threading.Thread.__init__(self)
    self.camera = camera
    self.shared = shared
    
  def run(self):
    cap = cv2.VideoCapture(self.camera)
    self.shared.width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    self.shared.height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

    while cap.isOpened() and self.shared.running:
      
        success, image = cap.read()
        if not success:
            continue
        
        self.shared.image = cv2.flip(image,1)
      
    cap.release()
