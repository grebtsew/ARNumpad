import cv2
import numpy as np
import numpad
import math

def add_image_horizontal(image, image_to_add):
  return np.vstack((image, image_to_add))

def add_image_vertical(image, image_to_add):
  return np.hstack((image, image_to_add))


def detect_clicked_button_show_click(all_image, shared, visualize_button=True, visualize_text=True):
  """
  ! require shared.landmarks, shared.width, shared.height !
  ! shared.numpad_buttons_3dpos_and_type shared.dynamic_button_size shared.thumb_range! 
  ! shared.is_open shared.clicked_pos_3d  shared.clicked_button!
  Calculates the euclidean_distance between thumb 3d pos with landmarks to see 
  if a button is pressed.
  
  The visualize the result.

  Result saved in:
  shared.clicked_button 
  shared.clicked_pos_3d 
  """

  if ( shared.landmarks is not None):
    
    thumb = VectorMessageObj_To3dVec(shared.landmarks[0]["landmark"][4], shared.width, shared.height)

    current_distance = math.inf
    current_name = ""
    current_pos = None

    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2

    for [v,name] in shared.numpad_buttons_3dpos_and_type:
     

      tmp = euclidean_distance(thumb, v)
      if tmp < shared.dynamic_button_size*shared.thumb_range and tmp < current_distance: # change this to make buttons easier to press
        current_name = name
        current_pos = v

    shared.clicked_pos_3d = current_pos
    shared.clicked_button = current_name

    # Show text at bottom!
    if visualize_text:
      if shared.is_open:
        cv2.putText(all_image, 
        "Pressed button: "+current_name,
          (50,int(shared.height-50)),
          font,
          fontScale,
          fontColor,
          lineType)

      if visualize_button and current_pos is not None:
        all_image = cv2.circle(all_image, (current_pos[0],current_pos[1]), shared.dynamic_button_size,(0,255,0), 4)
  return all_image


def show_hand_and_score(all_image, shared):
  """
  ! require shared.handedness shared.box !
  Show default classification result from mediapipe
  In upper corner of box!
  """
  if ( shared.handedness is not None):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2
    cv2.putText(all_image, 
    str(shared.handedness["classification"][0]["label"])+" "+ str(round(shared.handedness["classification"][0]["score"],3)),
      (shared.box[0],shared.box[1]),
      font,
        fontScale,
        fontColor,
          lineType)
  return all_image

def calculate_and_show_numpad( all_image, shared, show_thumb_range = False, hide_if_open=False, min_hand_size_limit=0):
  """
  ! Require if hide_if_open == true: shared.is_open, shared.size !
  This is a help function to sort result depending on if you want to show debug or result mode.
  hide_if_open set the mode!
  """
  if(hide_if_open):
    if shared.is_open and shared.size[0]*shared.size[1] > min_hand_size_limit:
     
      return calculate_and_show_numpad_func(all_image, shared, show_thumb_range)
    else:
      return all_image
  else:
    return calculate_and_show_numpad_func(all_image, shared, show_thumb_range)

def calculate_and_show_numpad_func(all_image, shared, show_thumb_range):
  """
  Executed function of the above
   ! Requires: shared.landmarks, shared.handedness, shared.leftNumpadEnum, !
   ! shared.thumb_range,  shared.rightNumpadEnum, !
   ! shared.dynamic_button_size !
   ! shared.size, shared.width, shared.height !
   ! shared.numpad_buttons_3dpos_and_type !
  Calculate and place numpad numbers with circles
  Depending on which hand is used. Mirror positions using numpad.Enums.
  Basicly the end result visualisation function!
  """

  shared.numpad_buttons_3dpos_and_type = []

  if shared.handedness is not None:
    # Right or Left hand?
  
    enum_type = None
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 1
    fontColor = (255,255,255)
    color = (0,0,255)
    lineType               = 2
    enum_arr = []

    if(shared.handedness["classification"][0]["label"] == "left"):
      enum_arr = shared.leftNumpadEnum
      enum_type = numpad.LeftNumpadPoints
    else:
      enum_arr = shared.rightNumpadEnum
      enum_type = numpad.RightNumpadPoints

    for index in enum_arr:
      #print(enum_type(index).value[0], len(shared.landmarks[0]["landmark"]))
      enum_name =  enum_type(index)
      #print(str(enum_name))
      number_name = str(enum_name).split(".")[1].replace("Key","")

       # Clamp size of circle
      dynamic_size = int(max(shared.size)/35)
      if dynamic_size <= 0:
        dynamic_size = 1

      pos = shared.filtered_landmarks[int(enum_type(index).value[0])]
      tmp =(int(pos[0]), int(pos[1]))

      if (number_name == "Thumb"): 
        color = (0,255,255)
        if show_thumb_range:
          all_image = cv2.circle(all_image, tmp, dynamic_size*shared.thumb_range,color, 2)
        color = (255,255,0)
      else:
        shared.numpad_buttons_3dpos_and_type.append([np.array([int(pos[0]), int(pos[1]), int(pos[2])]), number_name])
      
      pos = tmp
      """
      #print(enum_type(index),"at", pos)
      #print("circle dynamic size", )
      """
      shared.dynamic_button_size = dynamic_size

      all_image = cv2.circle(all_image, pos, dynamic_size,color, 2)

      # Show correct number
      cv2.putText(all_image, number_name,  (pos[0]-dynamic_size,pos[1]+dynamic_size), font, fontScale, fontColor, lineType)
  return all_image

def calculate_and_draw_center_of_hand(all_image, shared, visualize_point=True):
  """
  ! Require shared.landmarks, shared.width, shared.height !
  Calculate median of all landmarks
  Calculate kalman on midpoint
  Redistribute landmarks depending on kalman result
  Print resulting middle point
  
  Save result at:
  shared.midpoint
  shared.filtered_landmarks

  Note: several good prints saved in this code!
  """
  if shared.landmarks is not None:
    all_x_landmarks = []
    all_y_landmarks = []
    all_z_landmarks = []
    for landmark in shared.landmarks[0]["landmark"]:
      all_x_landmarks.append(landmark["x"])
      all_y_landmarks.append(landmark["y"])
      all_z_landmarks.append(landmark["z"])

    median_x = int(np.median(all_x_landmarks)* shared.width)
    median_y = int(np.median(all_y_landmarks)* shared.height)
    median_z = int(np.median(all_z_landmarks)* shared.height)

    radius = 2
    color = (255, 0, 0)
    thickness = 8

    """
    #before = []
    #for i in range(0,len(all_x_landmarks)):
    #  before.append([all_x_landmarks[i]* shared.width, all_y_landmarks[i]* shared.height, all_z_landmarks[i]* shared.height])
    """

    # Filter all landmarks
    diff_array= []
    for i in range(0,len(all_x_landmarks)):
      diff_array.append([all_x_landmarks[i]* shared.width-median_x, all_y_landmarks[i]* shared.height-median_y, all_z_landmarks[i]* shared.height-median_z])

    filtered = shared.kalman_filter.filter_pos(median_x, median_y)

    shared.midpoint = [filtered[0], filtered[1], median_z]

    # Place landmarks correct
    filtered_landmarks = []
    for i in range(0,len(all_x_landmarks)):
      filtered_landmarks.append([diff_array[i][0]+shared.midpoint[0],diff_array[i][1]+shared.midpoint[1], diff_array[i][2]+shared.midpoint[2]])

    """
    #print("before",before)  
    #print("after",filtered_landmarks)
    """

    shared.filtered_landmarks = filtered_landmarks
    if visualize_point:
      all_image = cv2.circle(all_image, (median_x, median_y), radius, color, thickness)
    
  return all_image

def calculate_hand_size_and_draw_boxes(all_image, shared, visualize_boxes=True):
  """
   ! Require shared.landmarks, shared.width, shared.height !
  Calculate max and min distance between landmarks to median center.
  Then multiply it by 2. 
  Store result as:
  shared.size = (x,y)
  shared.box = [xmin, ymin, xmax,ymax]
  """

  if shared.landmarks is not None:
    all_x_landmarks = []
    all_y_landmarks = []
    for landmark in shared.landmarks[0]["landmark"]:
      all_x_landmarks.append(landmark["x"])
      all_y_landmarks.append(landmark["y"])
    
    minx = min(all_x_landmarks)
    maxx = max(all_x_landmarks)
    miny = min(all_y_landmarks)
    maxy = max(all_y_landmarks)
    
    x_diff = (maxx - minx)/2
    y_diff = (maxy - miny)/2

    start_box = (int((minx-x_diff)*shared.width), int((miny-y_diff)*shared.height))
    end_box = (int((maxx+x_diff)*shared.width), int((maxy+y_diff)*shared.height))

    shared.size=(end_box[0]-start_box[0], end_box[1]-start_box[1])
    shared.box = [start_box[0], start_box[1], end_box[0], end_box[1]]

    if visualize_boxes:
      color = (0, 0, 255) 
      thickness = 2

      all_image = cv2.rectangle(all_image, start_box, end_box, color, thickness) 
  return all_image

def calculate_and_show_direction(all_image, shared, visualize_vectors=True):
  """
  ! Require shared.landmarks, shared.direction_scale, shared.width, shared.height !
  With simple Linear algebra we use 3 points on the hand to create a plane.
  Then calculate and print Normal Base Vectors (x,y,(z = n) for plane) of length "shared.direction_scale" see below 
  Save result as shared.rotmatrix
  """

  if shared.landmarks is not None:

    m0 =shared.landmarks[0]["landmark"][0] 
    m1 = shared.landmarks[0]["landmark"][5] 
    m2 = shared.landmarks[0]["landmark"][17] 
    
    p2 =  np.array([m2["x"],m2["y"],m2["z"]])
    p1 =  np.array([m1["x"],m1["y"],m1["z"]])
    p0 = np.array([m0["x"],m0["y"],m0["z"]])
    
    n0 = p0-p1
    n1 = p2- (np.dot(p2, n0)/np.sqrt(sum(n0**2))**2)*n0 # Linear Projection!
    n2 = np.cross(n0,n1)

    f0 = n0/np.linalg.norm(n0)
    f1 = n1/np.linalg.norm(n1)
    f2 = n2/np.linalg.norm(n2)

    #print("is ortogonal ", round(np.dot(f0, f1)), round(np.dot(f0, f2),4), round(np.dot(f1, f2),4))

    

    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2

    origo = (int(p0[0]*shared.width), int(p0[1]*shared.height))
    xaxis = (int(p0[0]*shared.width-f0[0]*shared.direction_scale), int(p0[1]*shared.height-f0[1]*shared.direction_scale))
    yaxis = (int(p0[0]*shared.width+f1[0]*shared.direction_scale), int(p0[1]*shared.height+f1[1]*shared.direction_scale))
    zaxis = (int(p0[0]*shared.width+f2[0]*shared.direction_scale), int(p0[1]*shared.height+f2[1]*shared.direction_scale))

    shared.rotmatrix = np.array([f0,f1,f2])
    
    if visualize_vectors:
      cv2.putText(all_image, "x",  xaxis, font, fontScale, fontColor, lineType)
      all_image = cv2.line(all_image, origo, xaxis , (255,0,0), 6)
      cv2.putText(all_image, "y", yaxis, font, fontScale, fontColor, lineType)
      all_image = cv2.line(all_image,origo ,yaxis, (0,255,0), 6)
      cv2.putText(all_image, "z", zaxis, font, fontScale, fontColor, lineType)
      all_image = cv2.line(all_image, origo, zaxis, (0,0,255), 6)

  return all_image

def VectorMessageObj_To2dVec(messageObj, width=1, height=1):
  return np.array([int(messageObj["x"]*width),int(messageObj["y"]*height)])

def VectorMessageObj_To3dVec(messageObj, width=1, height=1):
  return np.array([int(messageObj["x"]*width),int(messageObj["y"]*height), int(messageObj["z"]*height)])

def euclidean_distance(v1,v2): # np vectors!
  return np.linalg.norm(v1-v2)

def calculate_show_is_open(all_image,shared, visualize_text=True, visualize_detection=True):
  """
  ! Require shared.landmarks, shared.size, shared.width, shared.height !
  Calculates if hand is open
  Uses endpoints of fingers to see if they intersect the wrist point
  Using euclidean_distance.
  Save result as shared.is_open
  """
  # if landmark 8,12,16,20 is close to 0
  if shared.landmarks is not None:

    too_close_limit = int(max(shared.size)/3)
    _open = True

    intresting_landmarks = [
      VectorMessageObj_To3dVec(shared.landmarks[0]["landmark"][8], shared.width, shared.height),
      VectorMessageObj_To3dVec(shared.landmarks[0]["landmark"][12], shared.width, shared.height),
      VectorMessageObj_To3dVec(shared.landmarks[0]["landmark"][16], shared.width, shared.height),
      VectorMessageObj_To3dVec(shared.landmarks[0]["landmark"][20], shared.width, shared.height)
    ]

    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2
    wrist = VectorMessageObj_To3dVec(shared.landmarks[0]["landmark"][0], shared.width, shared.height)

    res = []
    for landmark in intresting_landmarks:
      tmp = euclidean_distance(wrist, landmark)
      res.append(tmp)
      if tmp < too_close_limit:
        _open = False
    
    shared.is_open = _open
    # show text
    if visualize_text:
      cv2.putText(all_image, 
      "Open: "+str(_open),
        (shared.box[0],shared.box[3]),
        font,
          fontScale,
          fontColor,
            lineType)
    
    if visualize_detection:
      # show lines! 
      origo = VectorMessageObj_To2dVec(shared.landmarks[0]["landmark"][0], shared.width, shared.height)

      # show close limit
      cv2.circle(all_image, (origo[0],origo[1]), too_close_limit, (255,0,0), 2)

      for i in range(0,len(res)):
        color = (0,0,255)
        if res[i]< too_close_limit:
          color = (0,255,0)
        all_image = cv2.line(all_image, (origo[0],origo[1]), (intresting_landmarks[i][0], intresting_landmarks[i][1]) , color, 1)

  return all_image
