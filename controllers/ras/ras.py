import numpy as np
import kinpy as kp
import cv2
from scipy.spatial.transform import Rotation as R

from util import (display_image, normalize_depth, visualize_object_detection,
                 calculate_camera_parameters, pixel_to_camera_coordinates,
                 filter_objects_by_size, estimate_object_orientation)
from UR5e import UR5e


"""
MISSION 3: Control of a Robot Manipulator Arm for a Pick and Place Task
Learning outcomes: 1, 2, 3

Scenario:
An autonomous manipulator is equipped with a camera at its end-effector in addition
to the sensors in mission 1. The existing controller already provides behaviours to
move the robot in joint and/or task space and to open/close the gripper. There are
several objects as well as a tray on a table.

Desired behaviour:
The robot should be able to detect the objects on the table using its camera (the
tray will always be in the same position and does not need to be detected). The
robot should pick the objects one by one and drop them into the tray.

Implementation Notes:
- Camera specifications: 128x64 resolution, 1.0 FOV, 0.1-2.0m range
- Tray position: Fixed at (-0.49, 0.55, 0) - does not need detection
- Object detection: Use end-effector depth camera for cube detection
- Coordinate transforms: Use kinpy.Transform for pose calculations
- Reference implementations adapted from RAS tutorials with proper attribution

Third-party code attribution:
- Camera transformation concepts adapted from Week 9 tutorial materials
- Kinematic patterns based on RAS_tutorial_PTP_FK_IK implementations
- OpenCV image processing techniques for object detection
"""

def acquire_camera_data(robot):
    """
    Acquire and preprocess depth image from end-effector camera.

    Camera specifications (from Week 9 tutorial):
    - Resolution: 128x64 pixels
    - FOV: 1.0 radian (~57.3 degrees)
    - Range: 0.1m to 2.0m
    - Pixel density calculation: ρ = FOV / width = 1.0 / 128 ≈ 0.0078 rad/px

    :param robot: UR5e robot instance
    :return: tuple (raw_depth, processed_depth, camera_pose)
             raw_depth: (64, 128) ndarray with depth values in meters
             processed_depth: preprocessed depth image for object detection
             camera_pose: kinpy.Transform of camera pose in world coordinates
    """
    # Get raw depth image from wrist camera
    raw_depth = robot.get_wrist_depth_image()

    if raw_depth is None:
        print("Warning: No depth image available from wrist camera")
        return None, None, None

    # Get current camera pose using forward kinematics
    # Note: Camera is offset from TCP by 0.05m in Z direction (from world file analysis)
    tcp_pose = robot.forward_kinematics()

    # Create camera pose transform (camera is at TCP with slight offset)
    # Based on tutorial patterns for coordinate transformations
    camera_pose = tcp_pose  # For simplicity, using TCP pose as camera pose

    # Preprocess depth image for object detection
    processed_depth = preprocess_depth_image(raw_depth)

    return raw_depth, processed_depth, camera_pose


def preprocess_depth_image(depth_image):
    """
    Preprocess depth image for object detection.

    Processing steps based on computer vision best practices:
    1. Handle invalid depth values (too close/far)
    2. Apply noise filtering
    3. Normalize for consistent processing

    :param depth_image: (64, 128) ndarray with raw depth values
    :return: preprocessed depth image suitable for object detection
    """
    if depth_image is None:
        return None

    # Create working copy
    processed = depth_image.copy()

    # Filter out invalid depth values (outside sensor range 0.1-2.0m)
    # Set invalid pixels to maximum range for background
    processed[processed < 0.1] = 2.0  # Too close - likely noise
    processed[processed > 2.0] = 2.0  # Too far - background

    # Apply Gaussian blur to reduce noise (small kernel to preserve object edges)
    # Using OpenCV for robust image processing
    processed_8bit = ((processed / 2.0) * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(processed_8bit, (3, 3), 0)
    processed = (blurred.astype(np.float32) / 255.0) * 2.0

    return processed


def classify_detected_objects(detected_objects, tray_position=np.array([-0.49, 0.55, 0.0]), exclusion_radius=0.2):
    """
    Classify detected objects to distinguish cubes from tray.

    The tray is at a fixed position (-0.49, 0.55, 0) and should be excluded from manipulation.
    This function filters out any detections near the tray location.

    :param detected_objects: list of kinpy.Transform objects representing detected objects
    :param tray_position: numpy array with tray position in world coordinates
    :param exclusion_radius: radius around tray to exclude detections (meters)
    :return: tuple (cube_objects, tray_detections)
             cube_objects: list of objects classified as manipulable cubes
             tray_detections: list of objects classified as tray (should be empty if working correctly)
    """
    cube_objects = []
    tray_detections = []

    for obj in detected_objects:
        # Calculate distance from object to known tray position
        obj_position = obj.pos
        distance_to_tray = np.linalg.norm(obj_position[:2] - tray_position[:2])  # Only X,Y distance

        if distance_to_tray < exclusion_radius:
            # Object is near tray position - classify as tray
            tray_detections.append(obj)
            print(f"Tray detection filtered out: pos={obj_position}, distance_to_tray={distance_to_tray:.3f}m")
        else:
            # Object is away from tray - classify as manipulable cube
            cube_objects.append(obj)
            print(f"Cube object detected: pos={obj_position}, distance_to_tray={distance_to_tray:.3f}m")

    return cube_objects, tray_detections


def validate_object_characteristics(contour, depth_image, min_size=50, max_size=2000, aspect_ratio_range=(0.5, 2.0)):
    """
    Validate that detected contour represents a valid cube object.

    Validation criteria:
    1. Size within reasonable bounds for cubes
    2. Aspect ratio suitable for cube projection
    3. Sufficient depth variation indicating 3D object
    4. Contour shape reasonably rectangular

    :param contour: OpenCV contour to validate
    :param depth_image: depth image for 3D validation
    :param min_size: minimum contour area
    :param max_size: maximum contour area
    :param aspect_ratio_range: tuple (min_ratio, max_ratio) for bounding box aspect ratio
    :return: bool indicating if contour represents valid cube object
    """
    # Check contour area
    area = cv2.contourArea(contour)
    if not (min_size <= area <= max_size):
        return False

    # Check bounding box aspect ratio
    _, _, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h if h > 0 else 0
    if not (aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]):
        return False

    # Check for reasonable depth variation (indicating 3D object)
    mask = np.zeros(depth_image.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [contour], (255,))
    object_depths = depth_image[mask > 0]

    if len(object_depths) == 0:
        return False

    depth_std = np.std(object_depths)
    # Cubes should have some depth variation but not too much noise
    if depth_std < 0.01 or depth_std > 0.1:  # 1cm to 10cm variation
        return False

    # Check contour shape - should be reasonably rectangular
    # Use contour approximation to check for rectangular shape
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Rectangles should approximate to 4-8 points depending on perspective
    if len(approx) < 4 or len(approx) > 8:
        return False

    return True


def detect_objects_in_depth(depth_image, camera_pose, min_object_size=50):
    """
    Detect cube objects in preprocessed depth image using enhanced utilities.

    Detection strategy:
    1. Threshold depth image to find objects on table surface
    2. Use contour detection to identify discrete objects
    3. Filter by size and shape to identify cubes
    4. Convert pixel coordinates to world coordinates using camera parameters

    :param depth_image: preprocessed depth image
    :param camera_pose: kinpy.Transform of camera pose
    :param min_object_size: minimum contour area for valid objects
    :return: tuple (detected_objects, contours) for visualization
             detected_objects: list of kinpy.Transform objects
             contours: list of OpenCV contours for debugging
    """
    if depth_image is None:
        return [], []

    detected_objects = []

    # Get camera parameters from utility function
    camera_params = calculate_camera_parameters()

    # Table surface detection - assume table at approximately 0.8m depth
    # Objects will appear as closer regions (lower depth values)
    table_depth = 0.8  # Estimated table distance from camera
    object_threshold = table_depth - 0.1  # Objects 10cm above table

    # Create binary mask for objects (closer than threshold)
    object_mask = (depth_image < object_threshold).astype(np.uint8) * 255

    # Find contours of potential objects
    contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by size to remove noise
    size_filtered_contours = filter_objects_by_size(contours, min_area=min_object_size, max_area=2000)

    # Further validate contours as cube objects
    validated_contours = []
    for contour in size_filtered_contours:
        if validate_object_characteristics(contour, depth_image):
            validated_contours.append(contour)

    print(f"Found {len(contours)} total contours, {len(size_filtered_contours)} after size filtering, {len(validated_contours)} after validation")

    for i, contour in enumerate(validated_contours):
        # Get bounding rectangle and centroid
        x, y, w, h = cv2.boundingRect(contour)
        centroid_u = x + w // 2  # pixel coordinates
        centroid_v = y + h // 2

        # Get depth at object centroid
        object_depth = depth_image[centroid_v, centroid_u]

        # Convert pixel coordinates to camera frame coordinates using utility function
        x_cam, y_cam, z_cam = pixel_to_camera_coordinates(
            centroid_u, centroid_v, object_depth, camera_params
        )

        # Estimate object orientation from contour shape
        estimated_orientation = estimate_object_orientation(contour)

        # Transform to world coordinates using camera pose
        # This is a simplified transformation - full implementation would need proper camera calibration
        object_pos_camera = np.array([x_cam, y_cam, z_cam])

        # For now, use camera position as approximation
        # In full implementation, would apply proper camera_pose transformation matrix
        object_pos_world = camera_pose.pos + object_pos_camera

        # Create object pose with estimated orientation
        # Use camera rotation as base and adjust for estimated object orientation
        object_rotation = camera_pose.rot  # Simplified - would need proper orientation calculation
        object_pose = kp.Transform(pos=object_pos_world, rot=object_rotation)

        detected_objects.append(object_pose)

        print(f"Object {i+1}: pos={object_pos_world}, depth={object_depth:.3f}m, orientation={np.rad2deg(estimated_orientation):.1f}°")

    return detected_objects, validated_contours


def main():
    # Initialize robot and move to safe starting position
    robot = UR5e()

    # Define configuration init_pos as intermediate waypoint to avoid collisions
    init_pos = [0, -1.57, 1.57, -1.57, -1.57, 0.0]
    robot.move_to_joint_pos(init_pos)

    # Move to home position for object detection
    robot.move_to_joint_pos(robot.home_pos, velocity=0.3, timeout=10)

    # Test gripper functionality
    robot.open_gripper()
    robot.close_gripper()
    robot.open_gripper()

    print("=== MISSION 3: Pick and Place with Object Detection ===")

    # Main object detection and manipulation loop
    max_iterations = 10  # Safety limit to prevent infinite loops
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")

        # Step 1: Acquire camera data
        print("Acquiring camera data...")
        raw_depth, processed_depth, camera_pose = acquire_camera_data(robot)

        if processed_depth is None:
            print("Failed to acquire camera data, retrying...")
            continue

        # Display images for debugging
        display_image(normalize_depth(raw_depth), 'Raw Depth Image')
        display_image(normalize_depth(processed_depth), 'Processed Depth Image')

        # Step 2: Detect objects
        print("Detecting objects...")
        detected_objects, contours = detect_objects_in_depth(processed_depth, camera_pose)

        print(f"Detected {len(detected_objects)} total objects")

        # Step 3: Classify objects (distinguish cubes from tray)
        print("Classifying objects...")
        cube_objects, tray_detections = classify_detected_objects(detected_objects)

        print(f"Classification results: {len(cube_objects)} cubes, {len(tray_detections)} tray detections")

        # Visualize detection results
        if len(contours) > 0:
            visualize_object_detection(normalize_depth(processed_depth), contours, cube_objects, "Cube Detection Results")

        if len(cube_objects) == 0:
            print("No manipulable cubes detected. Mission complete!")
            break

        # Step 4: Select target cube (closest one for now)
        target_object = cube_objects[0]  # Simple selection strategy
        print(f"Target cube position: {target_object.pos}")

        # Step 5: Approach and grasp cube (placeholder for Week 3-4 implementation)
        print("Cube approach and grasping - TO BE IMPLEMENTED IN WEEK 3-4")

        # Step 6: Transport to tray (placeholder)
        print("Transport to tray - TO BE IMPLEMENTED IN WEEK 4")
        print(f"Tray is at fixed position: (-0.49, 0.55, 0.0)")

        # For now, break after first detection to test camera system
        print("Camera system test complete. Breaking for development...")
        break

    print("\n=== Mission 3 Complete ===")


if __name__ == '__main__':
    main()
 