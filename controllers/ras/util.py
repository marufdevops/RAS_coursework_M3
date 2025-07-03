# YOU CAN MODIFY THIS FILE IF YOU LIKE.
# Enhanced utility functions for Mission 3 object detection and manipulation

import numpy as np
import cv2


def display_image(image, name, scale=2, wait=False):
    """
    function to display an image
    :param image: ndarray, the image to display
    :param name: string, a name for the window
    :param scale: int, optional, scaling factor for the image
    :param wait: bool, optional, if True, will wait for click/button to close window
    """
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, image.shape[1]*scale, image.shape[0]*scale)
    cv2.imshow(name, image)
    cv2.waitKey(0 if wait else 1)


def normalize_depth(depth_image):
    """
    function to normalize the depth image between 0 and 1 for better visualization.
    :param depth_image: 2d-array
    :returns: 2d-array, values between 0 and 1
    """
    if depth_image is None or np.allclose(depth_image, 0.0):
        return depth_image
    img = np.copy(depth_image)
    img *= 1.0/img.max()
    return img


def visualize_object_detection(depth_image, contours, detected_objects, window_name="Object Detection"):
    """
    Visualize object detection results on depth image.

    :param depth_image: normalized depth image for visualization
    :param contours: list of OpenCV contours found in image
    :param detected_objects: list of detected object poses
    :param window_name: name for display window
    """
    if depth_image is None:
        return

    # Convert depth image to 3-channel for color overlay
    vis_image = cv2.cvtColor((depth_image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Draw contours in green
    cv2.drawContours(vis_image, contours, -1, (0, 255, 0), 1)

    # Draw bounding boxes and centroids for detected objects
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) >= 50:  # Only show significant contours
            x, y, w, h = cv2.boundingRect(contour)

            # Draw bounding rectangle in blue
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (255, 0, 0), 1)

            # Draw centroid in red
            centroid_x = x + w // 2
            centroid_y = y + h // 2
            cv2.circle(vis_image, (centroid_x, centroid_y), 3, (0, 0, 255), -1)

            # Add object number label
            cv2.putText(vis_image, f"Obj{i+1}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    display_image(vis_image, window_name, scale=4)


def calculate_camera_parameters():
    """
    Calculate camera parameters based on specifications from Week 9 tutorial.

    Camera specifications:
    - Resolution: 128x64 pixels
    - FOV: 1.0 radian (~57.3 degrees)
    - Range: 0.1m to 2.0m

    :return: dict with camera parameters
    """
    return {
        'width': 128,
        'height': 64,
        'fov': 1.0,  # radians
        'fov_degrees': 57.3,  # degrees
        'pixel_density': 1.0 / 128,  # radians per pixel
        'min_range': 0.1,  # meters
        'max_range': 2.0,  # meters
        'center_u': 64,  # image center x
        'center_v': 32,  # image center y
    }


def pixel_to_camera_coordinates(pixel_u, pixel_v, depth, camera_params=None):
    """
    Convert pixel coordinates to 3D coordinates in camera frame.

    Based on camera transformation principles from Week 9 tutorial.

    :param pixel_u: pixel x coordinate
    :param pixel_v: pixel y coordinate
    :param depth: depth value at pixel location (meters)
    :param camera_params: camera parameter dict (optional)
    :return: tuple (x_cam, y_cam, z_cam) in camera frame coordinates
    """
    if camera_params is None:
        camera_params = calculate_camera_parameters()

    # Calculate angles from camera center
    angle_u = (pixel_u - camera_params['center_u']) * camera_params['pixel_density']
    angle_v = (pixel_v - camera_params['center_v']) * camera_params['pixel_density']

    # Convert to 3D coordinates in camera frame
    x_cam = depth * np.tan(angle_u)
    y_cam = depth * np.tan(angle_v)
    z_cam = depth

    return x_cam, y_cam, z_cam


def filter_objects_by_size(contours, min_area=50, max_area=2000):
    """
    Filter detected contours by area to remove noise and oversized detections.

    :param contours: list of OpenCV contours
    :param min_area: minimum contour area in pixels
    :param max_area: maximum contour area in pixels
    :return: list of filtered contours
    """
    filtered_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            filtered_contours.append(contour)

    return filtered_contours


def estimate_object_orientation(contour):
    """
    Estimate object orientation from contour shape.

    For cube objects, this provides a rough orientation estimate
    that can be refined during approach planning.

    :param contour: OpenCV contour
    :return: estimated orientation angle in radians
    """
    # Fit minimum area rectangle to get orientation
    rect = cv2.minAreaRect(contour)
    angle = rect[2]  # angle in degrees

    # Convert to radians and normalize
    angle_rad = np.deg2rad(angle)

    # Normalize to [0, π/2] for cube symmetry
    while angle_rad > np.pi/2:
        angle_rad -= np.pi/2
    while angle_rad < 0:
        angle_rad += np.pi/2

    return angle_rad


def create_synthetic_depth_scene(scene_type="standard", noise_level=0.01):
    """
    Create synthetic depth images for testing object detection accuracy.

    This function generates various test scenarios to validate detection
    robustness under different conditions.

    :param scene_type: str, type of scene to generate
                      "standard" - normal objects on table
                      "cluttered" - objects close together
                      "edge_cases" - objects near table edges
                      "noisy" - high noise conditions
    :param noise_level: float, amount of Gaussian noise to add
    :return: tuple (depth_image, ground_truth_objects)
             depth_image: (64, 128) synthetic depth image
             ground_truth_objects: list of known object positions for validation
    """
    # Initialize depth image with table surface at 0.8m
    depth_image = np.ones((64, 128), dtype=np.float32) * 0.8
    ground_truth_objects = []

    if scene_type == "standard":
        # Standard scene: 3 well-separated objects
        objects = [
            {"center": (40, 30), "size": (8, 8), "depth": 0.7},  # Object 1
            {"center": (80, 45), "size": (10, 10), "depth": 0.65},  # Object 2
            {"center": (60, 15), "size": (6, 6), "depth": 0.72},  # Object 3
        ]

    elif scene_type == "cluttered":
        # Cluttered scene: objects close together
        objects = [
            {"center": (50, 30), "size": (8, 8), "depth": 0.7},
            {"center": (58, 35), "size": (7, 7), "depth": 0.68},  # Close to first
            {"center": (45, 38), "size": (6, 6), "depth": 0.72},  # Close to first
            {"center": (90, 20), "size": (9, 9), "depth": 0.66},
        ]

    elif scene_type == "edge_cases":
        # Edge cases: objects near image boundaries
        objects = [
            {"center": (10, 10), "size": (8, 8), "depth": 0.7},   # Near corner
            {"center": (120, 30), "size": (6, 6), "depth": 0.68}, # Near right edge
            {"center": (60, 5), "size": (7, 7), "depth": 0.72},   # Near top edge
            {"center": (40, 58), "size": (8, 8), "depth": 0.69},  # Near bottom edge
        ]

    elif scene_type == "noisy":
        # Noisy conditions with higher noise level
        noise_level = 0.05  # Override with higher noise
        objects = [
            {"center": (50, 30), "size": (10, 10), "depth": 0.7},
            {"center": (80, 45), "size": (8, 8), "depth": 0.68},
        ]

    else:
        # Default to standard scene
        objects = [{"center": (64, 32), "size": (10, 10), "depth": 0.7}]

    # Add objects to depth image
    for obj in objects:
        center_x, center_y = obj["center"]
        size_x, size_y = obj["size"]
        depth = obj["depth"]

        # Create object region
        x_start = max(0, center_x - size_x // 2)
        x_end = min(128, center_x + size_x // 2)
        y_start = max(0, center_y - size_y // 2)
        y_end = min(64, center_y + size_y // 2)

        # Set object depth
        depth_image[y_start:y_end, x_start:x_end] = depth

        # Add to ground truth (convert to world coordinates approximately)
        camera_params = calculate_camera_parameters()
        x_world, y_world, z_world = pixel_to_camera_coordinates(center_x, center_y, depth, camera_params)
        ground_truth_objects.append({
            "position": np.array([x_world, y_world, z_world]),
            "pixel_center": (center_x, center_y),
            "depth": depth
        })

    # Add Gaussian noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, depth_image.shape)
        depth_image += noise
        # Clamp to valid depth range
        depth_image = np.clip(depth_image, 0.1, 2.0)

    return depth_image, ground_truth_objects


def evaluate_detection_accuracy(detected_objects, ground_truth_objects, position_threshold=0.1):
    """
    Evaluate object detection accuracy against ground truth.

    :param detected_objects: list of detected object poses
    :param ground_truth_objects: list of ground truth object data
    :param position_threshold: maximum distance for successful detection (meters)
    :return: dict with accuracy metrics
    """
    if len(ground_truth_objects) == 0:
        return {"precision": 0, "recall": 0, "f1_score": 0, "true_positives": 0, "false_positives": len(detected_objects), "false_negatives": 0}

    true_positives = 0
    matched_gt = set()

    # Find matches between detected and ground truth objects
    for detected in detected_objects:
        best_match_distance = float('inf')
        best_match_idx = -1

        for i, gt_obj in enumerate(ground_truth_objects):
            if i in matched_gt:
                continue

            # Calculate distance between detected and ground truth positions
            distance = np.linalg.norm(detected.pos - gt_obj["position"])

            if distance < position_threshold and distance < best_match_distance:
                best_match_distance = distance
                best_match_idx = i

        if best_match_idx >= 0:
            true_positives += 1
            matched_gt.add(best_match_idx)

    false_positives = len(detected_objects) - true_positives
    false_negatives = len(ground_truth_objects) - true_positives

    # Calculate metrics
    precision = true_positives / len(detected_objects) if len(detected_objects) > 0 else 0
    recall = true_positives / len(ground_truth_objects)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "total_detected": len(detected_objects),
        "total_ground_truth": len(ground_truth_objects)
    }


def run_detection_accuracy_tests():
    """
    Run comprehensive detection accuracy tests across multiple scenarios.

    :return: dict with test results for each scenario
    """
    test_scenarios = ["standard", "cluttered", "edge_cases", "noisy"]
    results = {}

    print("=== Object Detection Accuracy Testing ===")

    for scenario in test_scenarios:
        print(f"\nTesting scenario: {scenario}")

        # Create synthetic scene
        depth_image, ground_truth = create_synthetic_depth_scene(scenario)

        print(f"Ground truth: {len(ground_truth)} objects")
        for i, gt_obj in enumerate(ground_truth):
            print(f"  Object {i+1}: pixel=({gt_obj['pixel_center'][0]}, {gt_obj['pixel_center'][1]}), depth={gt_obj['depth']:.3f}m")

        # Note: Full detection testing would require importing the detection functions
        # This is a framework for testing that would be used within the Webots environment

        results[scenario] = {
            "ground_truth_count": len(ground_truth),
            "synthetic_scene_created": True,
            "depth_image_shape": depth_image.shape,
            "depth_range": (float(np.min(depth_image)), float(np.max(depth_image)))
        }

    return results
    
    