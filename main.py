# MediaPipe Hands
#
# Sources: 
# https://google.github.io/mediapipe/solutions/hands.html // original code 
# https://gist.github.com/TheJLifeX/74958cc59db477a91837244ff598ef4a // gesture detection
# https://github.com/Kazuhito00/mediapipe-python-sample // performance
# https://github.com/JuliaPoo/MultiHand-Tracking // left or right hand? Palm detection? 3d detection?
# https://towardsdatascience.com/handtrackjs-677c29c1d585 // more information, samples for .js

# Imported main
import cv2
import mediapipe as mp
from google.protobuf.json_format import MessageToDict # Convert google datatypes

# Own library
import calibrate as ca
import shared_variables as sv
import calculations as calc
import camera as cam

if __name__ == "__main__":

  # Select camera to use
  camera = 0

  # Setup parameters
  # Calculate appropriate detection confidence for camera used!
  # Protocol 1. show hand press enter, 2. show back of hand press enter.
  calibrated_detection_confidence = ca.calibrate(camera)

  # Setup shared variables between threads
  shared = sv.shared_variables()

  # initiate detection
  mp_drawing = mp.solutions.drawing_utils
  mp_hands = mp.solutions.hands

  hands = mp_hands.Hands(
      min_detection_confidence=calibrated_detection_confidence[1],
        min_tracking_confidence=0.99, # Nice visuals!
        max_num_hands=1 # default, Remove possibility of "hands in hands"
        ) 


  # Start image read thread
  cam.camera_handler(camera,shared).start()

  while shared.running:
    while shared.image is not None:
      # Preform hand detection
      image = shared.image
      image.flags.writeable = False
      results = hands.process(image)
      image.flags.writeable = True

      # Collect result from mediapipe, into format usable by default python
      if (results.multi_handedness is not None):
        for idx, hand_handedness in enumerate(results.multi_handedness):
          shared.handedness =  MessageToDict(hand_handedness)
      else:
        shared.handedness = None

      if(results.multi_hand_landmarks is not None):
        shared.landmarks = []
        for idx, multi_hand_landmarks in enumerate(results.multi_hand_landmarks):
          shared.landmarks.append(MessageToDict(multi_hand_landmarks))
      else:
        shared.landmarks = None
      
      #
      # Visualizations
      #

      default_image = image.copy()
      if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
          mp_drawing.draw_landmarks(
              default_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

      hand_image = image.copy()
      hand_image = calc.show_hand_and_score(hand_image, shared)

      scaling_filter_image = image.copy()
      scaling_filter_image = calc.calculate_and_draw_center_of_hand(scaling_filter_image, shared)
      scaling_filter_image = calc.calculate_hand_size_and_draw_boxes(scaling_filter_image, shared)

      directional_image = image.copy()
      directional_image = calc.calculate_and_show_direction(directional_image, shared)

      openClose_image = image.copy()
      openClose_image = calc.calculate_show_is_open(openClose_image, shared)

      all_image = image.copy()
      all_image = calc.calculate_and_draw_center_of_hand(all_image, shared)
      all_image = calc.calculate_hand_size_and_draw_boxes(all_image, shared)
      all_image = calc.calculate_and_show_direction(all_image, shared)
      all_image = calc.show_hand_and_score(all_image, shared)
      all_image = calc.calculate_show_is_open(all_image,shared)
      all_image = calc.calculate_and_show_numpad(all_image, shared, show_thumb_range=True)
      all_image = calc.detect_clicked_button_show_click(all_image, shared)

      result_image = image.copy()
      result_image = calc.calculate_and_show_numpad(result_image, shared, hide_if_open=True, min_hand_size_limit=50000) # hide numpad if hand is close or hand-size too small!
      result_image = calc.detect_clicked_button_show_click(result_image, shared)
      
      # Place all images in one for visualization!
      all_in_one_image1 = calc.add_image_vertical(image, default_image)
      all_in_one_image1 = calc.add_image_vertical(all_in_one_image1, hand_image)
      all_in_one_image1 = calc.add_image_vertical(all_in_one_image1, scaling_filter_image)
      all_in_one_image2 = calc.add_image_vertical(directional_image, openClose_image)
      all_in_one_image2 = calc.add_image_vertical(all_in_one_image2, all_image)
      all_in_one_image2 = calc.add_image_vertical(all_in_one_image2, result_image)
      all_in_one_image = calc.add_image_horizontal(all_in_one_image1, all_in_one_image2)
      cv2.imshow('All in one!', all_in_one_image)

      if cv2.waitKey(1) & 0xFF == ord('q'):
        shared.running = False
        break

  hands.close()
    

