"""
This file contains code for calibrating camera
Saving calibration and collecting old calibrations
"""

import pickle as pkl
import os
import threading
import cv2
import time
import mediapipe as mp

class Status():
    start_calculate = False
    stop = False
    result = (0, 0)

def calibrate(camera):
    """Calculate appropriate detection confidence for camera used!
     Protocol 1. show hand press enter, 2. show back of hand press enter. """

    calibration = None

    calibration = collect_if_exist()

    if calibration is not None:
        return calibration
    else:
      
        status = Status()

        thread = threading.Thread(target = fast_cam_cap, args = (camera, status, ))
        thread.start()
       
        # Start calibration routine:
        input("Hold hand in front of camera with palm clearly visible, then press enter with other hand!")
        status.start_calculate = True
        thread.join()
        status.start_calculate = False
        time.sleep(1)

        high = calculate_higher_accuracy(camera)

        time.sleep(1)
        thread2 = threading.Thread(target = fast_cam_cap, args = (camera, status, ))
        thread2.start()
        # Calculate lower accuracy
        input("Now flip the hand so that palm faces away from camera, then press enter again!")
        status.start_calculate = True
        thread2.join()
        time.sleep(1)
        # Calculate lower accuracy
        low = calculate_lower_accuracy(camera, high)

        # Save to file
        print("Save result",(high,low), "to file!")
        save_calibration((high,low))

        return (high,low)

def save_calibration(res):
     with open('./data/calibration.pkl', 'wb') as handle:
        pkl.dump(res, handle, protocol=pkl.HIGHEST_PROTOCOL)

def calculate_lower_accuracy(camera, high):
    confidence_low=high
    confidence_change_rate = 0.01
    current = confidence_low - confidence_change_rate
    amount_of_success = 10 # number of detections needed to pass
    amount_of_failure = 3
    _success = 1
    failure = 1

    # initiate
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    cap = cv2.VideoCapture(camera)

    while cap.isOpened() and _success < amount_of_success:
        if(failure == amount_of_failure):
            failure = 1
            current = current- confidence_change_rate
            print("Calibration calculation of Low at: ", current)
            
        
        hands = mp_hands.Hands(min_detection_confidence=current, min_tracking_confidence=0.5)

        success, image = cap.read()
        if not success:
            continue


        image.flags.writeable = False
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True

       
        if results.multi_hand_landmarks is not None:
            
            _success = 1
            failure+=1
        else:
            _success+=1
            failure = 1

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)


        cv2.imshow('Default', image)
        cv2.waitKey(1)

        hands.close()

    cap.release()
    cv2.destroyWindow("Default")
    print("Low rate calculated to", current)
    return current

def calculate_higher_accuracy(camera):
    confidence_high=1
    confidence_change_rate = 0.01
    current = confidence_high
    amount_of_success = 3 # number of detections in a row! to pass
    amount_of_failure = 3
    _success = 1
    failure = 1

    # initiate
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    cap = cv2.VideoCapture(camera)

    while cap.isOpened() and _success < amount_of_success:
        if(failure == amount_of_failure):
            failure = 1
            current = current- confidence_change_rate
            print("Calibration calculation of High at: ", current)
            
        
        hands = mp_hands.Hands(min_detection_confidence=current, min_tracking_confidence=0.5)

        success, image = cap.read()
        if not success:
            continue


        image.flags.writeable = False
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True

       
        if results.multi_hand_landmarks is not None:
            #print("success",_success)
            _success+=1
            failure = 1
        else:
            _success = 1
            failure+=1

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)


        cv2.imshow('Default', image)
        cv2.waitKey(1)

        hands.close()

    cap.release()
    cv2.destroyWindow("Default")
    print("High rate calculated to", current)
    return current

def fast_cam_cap(camera, status):
    cap = cv2.VideoCapture(camera)
    while cap.isOpened() and not status.start_calculate:
        success, image = cap.read()
        if not success:
            continue
        
        cv2.imshow('Default', image)
        cv2.waitKey(1)
    cap.release()
    cv2.destroyWindow("Default")

def collect_if_exist(path="./data/calibration."):
    calibration_values = ()

    try:
        with open('./data/calibration.pkl', 'rb') as fp:
            calibration_values = pkl.load(fp) 
    except (FileNotFoundError, EOFError): 
        if (os.path.isdir('./data/')):
            return None
        else:
            os.mkdir('./data/')
            return None
   

    print("Collected calibration : ", calibration_values)
    return calibration_values



# Debug testing
#print("calibration", calibrate(0))