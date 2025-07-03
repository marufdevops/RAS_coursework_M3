#!/usr/bin/env python3

import kinpy as kp
import os

URDF_FN = '../../resources/ur5e_2f85_camera.urdf'

def test_urdf_loading():
    print(f"Current working directory: {os.getcwd()}")
    print(f"URDF file path: {URDF_FN}")
    print(f"URDF file exists: {os.path.exists(URDF_FN)}")
    
    if os.path.exists(URDF_FN):
        print(f"URDF file size: {os.path.getsize(URDF_FN)} bytes")
        
        # Try reading the file
        try:
            with open(URDF_FN, 'r') as f:
                urdf_content = f.read()
            print(f"Successfully read URDF content, length: {len(urdf_content)}")
            print(f"Content type: {type(urdf_content)}")
            print(f"First 100 characters: {urdf_content[:100]}")
            
            # Try building the chain
            try:
                chain = kp.build_serial_chain_from_urdf(urdf_content, 'TCP')
                print("Successfully built kinematic chain!")
                print(f"Chain: {chain}")
            except Exception as e:
                print(f"Error building chain: {e}")
                print(f"Error type: {type(e)}")
                
                # Try with different end effector
                try:
                    chain = kp.build_serial_chain_from_urdf(urdf_content, 'wrist_3_link')
                    print("Successfully built chain with wrist_3_link!")
                except Exception as e2:
                    print(f"Error with wrist_3_link: {e2}")
                    
        except Exception as e:
            print(f"Error reading URDF file: {e}")
    else:
        print("URDF file not found!")

if __name__ == '__main__':
    test_urdf_loading()
