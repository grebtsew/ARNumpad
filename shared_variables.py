import numpad
import kalman as kal

class shared_variables():

    handedness = dict()
    landmarks = []
    running = True
    width = None
    height = None

    #button smoothing parameters
    clicked_button = ""
    clicked_pos_3d = None
    

    direction_scale = 100 # length of x,y,z of direction

    thumb_range = 3
    is_open = False

    size = ()
    rotmatrix = None
    box = [0,0,0,0]
    image = None
    dynamic_button_size = 1
    numpad_buttons_3dpos_and_type = None

    def __init__(self):
        self.kalman_filter = kal.kalman()
        self.leftNumpadEnum = list(map(numpad.LeftNumpadPoints, numpad.LeftNumpadPoints))
        self.rightNumpadEnum = list(map(numpad.RightNumpadPoints, numpad.RightNumpadPoints))
        self.NumpadEnum = list(map(numpad.NumpadValues, numpad.NumpadValues))