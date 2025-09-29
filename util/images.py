class Images:
    MOVEMENT_IMAGES_A = [
        "movement_library/EA/Index_flexion_M1.png",
        "movement_library/EA/Index_Extension_M2.png",
        "movement_library/EA/Middle_Flexion_M3.png",
        "movement_library/EA/Middle_Extension_M4.png",
        "movement_library/EA/Ring_Flexion_M5.png",
        "movement_library/EA/Ring_Extension_M6.png",
        "movement_library/EA/Little_Flexion_M7.png",
        "movement_library/EA/Little_Extension_M8.png",
        "movement_library/EA/Thurmb_Adduction_M9.png",
        "movement_library/EA/Thurmb_Abduction_M10.png",
        "movement_library/EA/Thurmb_Flexion_M11.png",
        "movement_library/EA/Thurmb_Extension_M12.png"
    ]
    MOVEMENT_IMAGES_B = [
        "movement_library/EB/Thrumb_up_M13.png",
        "movement_library/EB/Extension_of_index_and_middle_M14.PNG.png",
        "movement_library/EB/Flexion_of_little_and_ring_M15.PNG.png",
        "movement_library/EB/Thumb_opposing_of base_of_little_finger_M16.PNG.png",
        "movement_library/EB/hands_open_M17.PNG.png",
        "movement_library/EB/Fingures_fixed_together_in_fist_M18.PNG.png",
        "movement_library/EB/pointing_index_M19.PNG.png",
        "movement_library/EB/adduction_of_extended_fingers_M20.PNG.png",
        "movement_library/EB/wrist_supination_middile_finger_M21.PNG.png",
        "movement_library/EB/wrist_pronation_M22.PNG.png",
        "movement_library/EB/wrist_supination_little_finger_M23.PNG.png",
        "movement_library/EB/wrist_pronation_little_finger_M24.PNG.png",
        "movement_library/EB/wrist_flexion_M25.PNG.png",
        "movement_library/EB/wrist_extension_M26.PNG.png",
        "movement_library/EB/wrist_radial_deviation_M27.PNG.png",
        "movement_library/EB/wrist_ular_deviation_M28.PNG.png",
        "movement_library/EB/wrist_extension_with_closed_hand_M29.PNG.png"
    ]

    REST = "movement_library/Rest_M0.png"

    NAME_TO_FILE = {
        # Rest
        "Rest": REST,

        # Set A
        "Index Flexion": "movement_library/EA/Index_flexion_M1.png",
        "Index Extension": "movement_library/EA/Index_Extension_M2.png",
        "Middle Flexion": "movement_library/EA/Middle_Flexion_M3.png",
        "Middle Extension": "movement_library/EA/Middle_Extension_M4.png",
        "Ring Flexion": "movement_library/EA/Ring_Flexion_M5.png",
        "Ring Extension": "movement_library/EA/Ring_Extension_M6.png",
        "Little Flexion": "movement_library/EA/Little_Flexion_M7.png",
        "Little Extension": "movement_library/EA/Little_Extension_M8.png",
        "Thumb Adduction": "movement_library/EA/Thurmb_Adduction_M9.png",
        "Thumb Abduction": "movement_library/EA/Thurmb_Abduction_M10.png",
        "Thumb Flexion": "movement_library/EA/Thurmb_Flexion_M11.png",
        "Thumb Extension": "movement_library/EA/Thurmb_Extension_M12.png",

        # Set B
        "Thumb Up": "movement_library/EB/Thrumb_up_M13.png",
        "Index and Middle Extension": "movement_library/EB/Extension_of_index_and_middle_M14.PNG.png",
        "Little and Ring Flexion": "movement_library/EB/Flexion_of_little_and_ring_M15.PNG.png",
        "Thumb Opposing Little Finger": "movement_library/EB/Thumb_opposing_of base_of_little_finger_M16.PNG.png",
        "Hands Open": "movement_library/EB/hands_open_M17.PNG.png",
        "Fingers Together in Fist": "movement_library/EB/Fingures_fixed_together_in_fist_M18.PNG.png",
        "Index Pointing": "movement_library/EB/pointing_index_M19.PNG.png",
        "Finger Adduction": "movement_library/EB/adduction_of_extended_fingers_M20.PNG.png",
        "Wrist Supination (Middle Finger)": "movement_library/EB/wrist_supination_middile_finger_M21.PNG.png",
        "Wrist Pronation": "movement_library/EB/wrist_pronation_M22.PNG.png",
        "Wrist Supination (Little Finger)": "movement_library/EB/wrist_supination_little_finger_M23.PNG.png",
        "Wrist Pronation (Little Finger)": "movement_library/EB/wrist_pronation_little_finger_M24.PNG.png",
        "Wrist Flexion": "movement_library/EB/wrist_flexion_M25.PNG.png",
        "Wrist Extension": "movement_library/EB/wrist_extension_M26.PNG.png",
        "Wrist Radial Deviation": "movement_library/EB/wrist_radial_deviation_M27.PNG.png",
        "Wrist Ulnar Deviation": "movement_library/EB/wrist_ular_deviation_M28.PNG.png",
        "Wrist Extension (Closed Hand)": "movement_library/EB/wrist_extension_with_closed_hand_M29.PNG.png",
    }

    FILE_TO_NAME = {v: k for k, v in NAME_TO_FILE.items()}

    MOVEMENT_TUPLES = list(NAME_TO_FILE.items())