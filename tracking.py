import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import replace
from scipy.optimize import linear_sum_assignment
import cv2
from data_structures import Region, TrackedObject


class KalmanTracker:
    def __init__(self, initial_position: Tuple[float, float]):
        self.kf = cv2.KalmanFilter(4, 2)
        # x' = x + vx*dt, y' = y + vy*dt
        dt = 1.0  # Time step (1 frame)
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0
        self.kf.statePost = np.array([
            [initial_position[0]],
            [initial_position[1]],
            [0.0],
            [0.0]
        ], dtype=np.float32)
    
    def predict(self) -> Tuple[float, float]:
        prediction = self.kf.predict()
        return (float(prediction[0]), float(prediction[1]))
    
    def update(self, measurement: Tuple[float, float]) -> Tuple[float, float]:
        meas = np.array([[measurement[0]], [measurement[1]]], dtype=np.float32)
        corrected = self.kf.correct(meas)
        return (float(corrected[0]), float(corrected[1]))
    
    def get_velocity(self) -> Tuple[float, float]:
        state = self.kf.statePost
        return (float(state[2]), float(state[3]))


class Tracker:
    def __init__(self, config: dict):
        self.config = config
        # Extract tracking parameters
        tracking_config = config.get('tracking', {})
        self.max_distance = tracking_config.get('max_distance', 100.0)
        self.max_track_age = config.get('max_track_age', 10)
        self.use_kalman = config.get('use_kalman', False)
        self.min_trajectory_length = tracking_config.get('min_trajectory_length', 3)
        self.history_buffer_size = tracking_config.get('history_buffer_size', 5)
        self.velocity_weight = tracking_config.get('velocity_weight', 0.3)
        
        # Tracking state
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.lost_tracks: Dict[int, TrackedObject] = {}
        self.exited_tracks: List[TrackedObject] = []
        
        # ID management
        self.next_id = 0
        self.id_pool: List[int] = []
        
        # Kalman filters
        self.kalman_filters: Dict[int, KalmanTracker] = {}
        
        # History for correction
        self.region_history: List[Tuple[int, List[Region]]] = []  # (frame_idx, regions)
        
        # Statistics
        self.total_objects_tracked = 0
        self.id_switches = 0
    
    def _get_next_id(self) -> int:
        if self.id_pool:
            return self.id_pool.pop(0)
        else:
            new_id = self.next_id
            self.next_id += 1
            self.total_objects_tracked += 1
            return new_id
    
    def _return_id(self, obj_id: int):
        if obj_id not in self.id_pool:
            self.id_pool.append(obj_id)
    
    def _compute_cost_matrix(
        self, 
        regions: List[Region], 
        frame_index: int
    ) -> np.ndarray:
        n_tracks = len(self.tracked_objects)
        n_regions = len(regions)
        
        if n_tracks == 0 or n_regions == 0:
            return np.array([])
        
        cost_matrix = np.zeros((n_tracks, n_regions))
        track_ids = list(self.tracked_objects.keys())
        
        for i, track_id in enumerate(track_ids):
            obj = self.tracked_objects[track_id]
            
            # Get predicted position
            if self.use_kalman and track_id in self.kalman_filters:
                predicted_pos = self.kalman_filters[track_id].predict()
            else:
                # Simple velocity-based prediction
                if len(obj.trajectory) >= 2:
                    last_frame, last_pos = obj.trajectory[-1]
                    prev_frame, prev_pos = obj.trajectory[-2]
                    frames_diff = last_frame - prev_frame
                    if frames_diff > 0:
                        vx = (last_pos[0] - prev_pos[0]) / frames_diff
                        vy = (last_pos[1] - prev_pos[1]) / frames_diff
                        frames_elapsed = frame_index - last_frame
                        predicted_pos = (
                            last_pos[0] + vx * frames_elapsed,
                            last_pos[1] + vy * frames_elapsed
                        )
                    else:
                        predicted_pos = last_pos
                else:
                    predicted_pos = obj.centroid
            
            # Compute cost for each region
            for j, region in enumerate(regions):
                # Euclidean distance
                dx = region.centroid[0] - predicted_pos[0]
                dy = region.centroid[1] - predicted_pos[1]
                distance = np.sqrt(dx * dx + dy * dy)
                
                velocity_cost = 0.0
                if len(obj.trajectory) >= 2 and self.velocity_weight > 0:
                    last_frame, last_pos = obj.trajectory[-1]
                    # Expected displacement
                    expected_dx = predicted_pos[0] - last_pos[0]
                    expected_dy = predicted_pos[1] - last_pos[1]
                    # Actual displacement
                    actual_dx = region.centroid[0] - last_pos[0]
                    actual_dy = region.centroid[1] - last_pos[1]
                    # Velocity difference
                    velocity_cost = np.sqrt(
                        (actual_dx - expected_dx) ** 2 + 
                        (actual_dy - expected_dy) ** 2
                    )
                # Combined cost
                cost_matrix[i, j] = distance + self.velocity_weight * velocity_cost
        
        return cost_matrix
    
    def _match_regions_to_tracks(
        self, 
        regions: List[Region],
        frame_index: int
    ) -> Tuple[Dict[int, int], List[int]]:
        if not self.tracked_objects or not regions:
            return {}, list(range(len(regions)))
        
        cost_matrix = self._compute_cost_matrix(regions, frame_index)
        
        if cost_matrix.size == 0:
            return {}, list(range(len(regions)))
    
        track_indices, region_indices = linear_sum_assignment(cost_matrix)
        matched_pairs = {}
        matched_region_set = set()
        track_ids = list(self.tracked_objects.keys())
        
        for track_idx, region_idx in zip(track_indices, region_indices):
            cost = cost_matrix[track_idx, region_idx]
            if cost < self.max_distance:
                track_id = track_ids[track_idx]
                matched_pairs[track_id] = region_idx
                matched_region_set.add(region_idx)
        unmatched_regions = [i for i in range(len(regions)) if i not in matched_region_set]
        
        return matched_pairs, unmatched_regions
    
    def _check_history_for_reappearance(
        self, 
        region: Region, 
        frame_index: int
    ) -> Optional[int]:
        if not self.lost_tracks:
            return None
        
        best_match_id = None
        best_distance = self.max_distance
        
        for track_id, lost_obj in self.lost_tracks.items():
            # Check how long it's been lost
            last_seen_frame = lost_obj.trajectory[-1][0]
            frames_lost = frame_index - last_seen_frame
            
            # Only consider if lost recently (within history buffer)
            if frames_lost <= self.history_buffer_size:
                last_pos = lost_obj.trajectory[-1][1]
                dx = region.centroid[0] - last_pos[0]
                dy = region.centroid[1] - last_pos[1]
                distance = np.sqrt(dx * dx + dy * dy)
                max_allowed = self.max_distance * (1 + frames_lost * 0.2)
                
                if distance < max_allowed and distance < best_distance:
                    best_distance = distance
                    best_match_id = track_id
        
        return best_match_id
    
    def _create_new_track(
        self, 
        region: Region, 
        frame_index: int, 
        timestamp: float
    ) -> TrackedObject:
        obj_id = self._get_next_id()
        
        obj = TrackedObject(
            id=obj_id,
            bbox=region.bbox,
            centroid=region.centroid,
            trajectory=[(frame_index, region.centroid)],
            Fr0=frame_index,  # Entry frame
            FrN=None,  # Not exited yet
            captured_img_ref=None,
            speed_m_s=None,
            metadata={'area': region.area, 'first_seen': timestamp}
        )
        if self.use_kalman:
            self.kalman_filters[obj_id] = KalmanTracker(region.centroid)
        
        return obj
    
    def _update_track(
        self, 
        obj: TrackedObject, 
        region: Region, 
        frame_index: int, 
        timestamp: float
    ):
        if self.use_kalman and obj.id in self.kalman_filters:
            kf = self.kalman_filters[obj.id]
            smoothed_pos = kf.update(region.centroid)
            obj.centroid = smoothed_pos
        else:
            obj.centroid = region.centroid
        
        obj.bbox = region.bbox
        
        obj.trajectory.append((frame_index, obj.centroid))
        
        obj.metadata['area'] = region.area
        obj.metadata['last_seen'] = timestamp
    
    def _mark_as_lost(self, track_id: int, frame_index: int):
        if track_id in self.tracked_objects:
            obj = self.tracked_objects[track_id]
            obj.metadata['frames_since_seen'] = obj.metadata.get('frames_since_seen', 0) + 1
            obj.metadata['last_active_frame'] = frame_index - 1
    
    def _cleanup_lost_tracks(self, frame_index: int):
        to_remove = []
        
        for track_id, obj in self.tracked_objects.items():
            frames_since_seen = obj.metadata.get('frames_since_seen', 0)
            
            if frames_since_seen > 0:
                # Move to lost tracks
                if track_id not in self.lost_tracks:
                    self.lost_tracks[track_id] = obj
                
                # If lost too long, mark as exited
                if frames_since_seen >= self.max_track_age:
                    # Set exit frame
                    obj.FrN = obj.metadata.get('last_active_frame', frame_index - self.max_track_age)
                    
                    # Move to exited tracks
                    self.exited_tracks.append(obj)
                    
                    # Clean up
                    to_remove.append(track_id)
                    if track_id in self.lost_tracks:
                        del self.lost_tracks[track_id]
                    if track_id in self.kalman_filters:
                        del self.kalman_filters[track_id]
                    
                    # Return ID to pool
                    self._return_id(track_id)
        
        # Remove from active tracks
        for track_id in to_remove:
            if track_id in self.tracked_objects:
                del self.tracked_objects[track_id]
    
    def _restore_lost_track(self, track_id: int, region: Region, frame_index: int, timestamp: float):
        obj = self.lost_tracks[track_id]
        self._update_track(obj, region, frame_index, timestamp)
        obj.metadata['frames_since_seen'] = 0
        self.tracked_objects[track_id] = obj
        del self.lost_tracks[track_id]
    
    def update(
        self, 
        regions: List[Region], 
        frame_index: int, 
        timestamp: float
    ) -> List[TrackedObject]:
        # Store in history buffer
        self.region_history.append((frame_index, regions))
        if len(self.region_history) > self.history_buffer_size:
            self.region_history.pop(0)
        
        # Match regions to existing tracks
        matched_pairs, unmatched_regions = self._match_regions_to_tracks(regions, frame_index)
        
        # Update matched tracks
        matched_track_ids = set()
        for track_id, region_idx in matched_pairs.items():
            region = regions[region_idx]
            obj = self.tracked_objects[track_id]
            self._update_track(obj, region, frame_index, timestamp)
            obj.metadata['frames_since_seen'] = 0  # Reset lost counter
            matched_track_ids.add(track_id)
        
        # Mark unmatched tracks as lost
        for track_id in list(self.tracked_objects.keys()):
            if track_id not in matched_track_ids:
                self._mark_as_lost(track_id, frame_index)
        
        # Handle unmatched regions (potential new objects or reappearances)
        for region_idx in unmatched_regions:
            region = regions[region_idx]
            
            # Check if this might be a reappeared lost track
            reappeared_id = self._check_history_for_reappearance(region, frame_index)
            
            if reappeared_id is not None:
                # Restore lost track
                self._restore_lost_track(reappeared_id, region, frame_index, timestamp)
            else:
                # Create new track
                new_obj = self._create_new_track(region, frame_index, timestamp)
                self.tracked_objects[new_obj.id] = new_obj
        
        # Cleanup tracks that have been lost too long
        self._cleanup_lost_tracks(frame_index)
        
        return self.get_active()
    
    def get_active(self) -> List[TrackedObject]:
        return list(self.tracked_objects.values())
    
    def get_exited(self) -> List[TrackedObject]:
        return self.exited_tracks.copy()
    
    def get_track_by_id(self, track_id: int) -> Optional[TrackedObject]:
        if track_id in self.tracked_objects:
            return self.tracked_objects[track_id]
        for obj in self.exited_tracks:
            if obj.id == track_id:
                return obj
        return None
    
    def get_statistics(self) -> dict:
        return {
            'total_objects_tracked': self.total_objects_tracked,
            'active_tracks': len(self.tracked_objects),
            'lost_tracks': len(self.lost_tracks),
            'exited_tracks': len(self.exited_tracks),
            'id_pool_size': len(self.id_pool),
            'next_id': self.next_id,
            'id_switches': self.id_switches
        }
    
    def reset(self):
        self.tracked_objects.clear()
        self.lost_tracks.clear()
        self.exited_tracks.clear()
        self.id_pool.clear()
        self.kalman_filters.clear()
        self.region_history.clear()
        self.next_id = 0
        self.total_objects_tracked = 0
        self.id_switches = 0
