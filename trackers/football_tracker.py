"""
Advanced Football Player Tracking using ByteTrack Algorithm
Dependencies: pip install supervision ultralytics numpy scipy opencv-python
"""

import cv2
import numpy as np
import supervision as sv
import os
from pathlib import Path
import pickle

# Import from local modules
from trackers.detector import (CLASS_IDS, HUMAN_CLASSES,
                               HUMAN_ACCEPT_CONF, create_detector)
from trackers.bbox_utils import get_center_of_bbox, get_bbox_width, get_foot_position

# tracks[] key for each detector class. Goalkeepers are kept separate from
# players so team assignment can exclude them (their kit deliberately differs
# from their own team's) -- see docs/archive/TODO_legacy.md section 5. The
# three human roles stay distinct end to end; none is ever folded into another.
TRACK_KEY = {
    'player': 'players',
    'goalkeeper': 'goalkeepers',
    'referee': 'referees',
    'ball': 'ball',
}
TRACK_KEYS = list(TRACK_KEY.values())
CLASS_ID_TO_KEY = {CLASS_IDS[name]: key for name, key in TRACK_KEY.items()}
# Only these reach ByteTrack. The ball is excluded by construction rather than
# by a downstream filter, so a future edit cannot silently readmit it.
HUMAN_TRACK_KEY = {CLASS_IDS[name]: TRACK_KEY[name] for name in HUMAN_CLASSES}


class FootballTracker:
    """
    Comprehensive football video analysis tracker based on ByteTrack
    Tracks players, goalkeepers, referees and the ball as distinct classes
    """

    def __init__(self, model_path='yolov8s.pt',
                use_roboflow=False,
                api_key=None,
                persist_cache=True,
                cache_dir='tracker_cache',
                confidence=0.25,
                imgsz=960,
                max_ball_gap=15,
                human_candidate_pool=False,
                ball_candidate_pool=False,
                detector=None,
                tracker_backend='cbiou',
                frame_rate=30.0):
        """
        Initialize the football tracker

        Args:
            model_path: Path to the local YOLO model (the production detector)
            use_roboflow: Opt in to the hosted Roboflow detector (labelling/benchmark only)
            api_key: Roboflow API key; falls back to ROBOFLOW_API_KEY
            persist_cache: Whether to save cache to disk
            cache_dir: Directory for cache files
            confidence: Detection confidence threshold
            imgsz: Inference image size for the local detector
            max_ball_gap: How many consecutive frames the last known ball box may
                be held for while the ball is undetected. After that the ball is
                reported as unknown rather than frozen in place. Only used when
                the pipeline does NOT run BallTemporalSelector; the selector
                supersedes this hold with a bounded, provenance-tagged decision.
            ball_candidate_pool: emit ball detections down to the rescue floor
                (0.10) tagged state='candidate_low_conf'. They are never
                accepted as observations here -- they are collected per frame
                as `ball_candidates` for BallTemporalSelector to adjudicate.
            detector: Pre-built detector to use instead of constructing one.
                Only for tests, which must not load real YOLO weights.
            tracker_backend: which association implementation to use.
                'cbiou'  -- vendored Roboflow CBIoUTracker 2.6.0 at its exact
                            library defaults, selected by T2, qualified by
                            integration, and the DEFAULT for every entry point.
                'legacy' -- supervision sv.ByteTrack(), the previously deployed
                            tracker, kept for rollback and regression comparison.
                Neither is tuned. The two differ only in association; detection,
                ball handling and role semantics are identical either way.
                Tests that assert legacy-specific association behaviour pin
                tracker_backend='legacy' explicitly rather than relying on the
                default, so the default can move without rewriting them.
            frame_rate: the frame rate of the stream the TRACKER sees. When the
                pipeline skips frames this is the effective rate, not the source
                rate -- a tracker told 30 fps while receiving 10 mis-scales
                every motion prediction it makes.
        """
        self.model_path = model_path
        self.max_ball_gap = max_ball_gap
        # The boundary between association evidence and public output. Every
        # tracked box below this is used to keep an identity alive and is then
        # withheld from tracks[], so reports and visualisation are unchanged.
        self.human_accept_conf = max(confidence, HUMAN_ACCEPT_CONF)
        self.human_candidate_pool = human_candidate_pool
        self.ball_candidate_pool = ball_candidate_pool
        self.use_roboflow = use_roboflow
        self.api_key = api_key
        self.persist_cache = persist_cache
        self.cache_dir = Path(cache_dir)

        if persist_cache:
            os.makedirs(self.cache_dir, exist_ok=True)

        self.detector = detector if detector is not None else create_detector(
            model_path=model_path,
            use_roboflow=use_roboflow,
            api_key=api_key,
            confidence=confidence,
            imgsz=imgsz,
            ball_candidate_pool=ball_candidate_pool,
            human_candidate_pool=human_candidate_pool,
        )

        # Association backend. CBIoU is vendored under rf_trackers rather than
        # installed as `trackers`, because EyeCU owns that name; see
        # rf_trackers/VENDOR_PROVENANCE.json.
        if tracker_backend not in ('legacy', 'cbiou'):
            raise ValueError(f'unknown tracker_backend {tracker_backend!r}')
        self.tracker_backend = tracker_backend
        self.frame_rate = float(frame_rate)
        if tracker_backend == 'cbiou':
            from rf_trackers import CBIoUTracker
            # exact T2 configuration: every parameter is a library default and
            # only the frame rate, which the API asks for, is supplied
            self.tracker = CBIoUTracker(frame_rate=self.frame_rate)
        else:
            self.tracker = sv.ByteTrack()

        # Populated by get_object_tracks(); read by the pipeline's final report.
        self.tracks = None
        # Per-frame ball candidates, aligned with tracks['ball'] by index.
        # None means "not available" -- e.g. tracks were restored from a cache
        # written before candidates were recorded. Consumers must treat that as
        # missing evidence, not as an empty candidate list.
        self.ball_candidates = None

        # Color map for visualization
        self.colors = {
            'player': (0, 255, 0),      # Green
            'goalkeeper': (0, 165, 255),  # Orange
            'ball': (0, 0, 255),        # Red
            'referee': (255, 255, 0),   # Yellow
            'team1': (255, 50, 50),     # Light red
            'team2': (50, 50, 255)      # Light blue
        }
        
    def add_position_to_tracks(self, tracks):
        """
        Add position information to tracked objects
        
        Args:
            tracks: Dictionary of tracked objects
        """
        for object_type, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object_type == 'ball':
                        # For ball, use center point
                        position = get_center_of_bbox(bbox)
                    else:
                        # For players and referees, use foot position (bottom center)
                        position = get_foot_position(bbox)
                    
                    tracks[object_type][frame_num][track_id]['position'] = position
    
    def apply_ball_temporal_selection(self, ball_positions, candidates=None,
                                      fps=None, frame_width=None, cuts=None):
        """
        Resolve the ball track with BallTemporalSelector. The authoritative
        ball post-processing step.

        This replaced `interpolate_ball_positions()`, which was removed. That
        method substituted [0,0,0,0] for a missing ball and then ran
        interpolate().bfill().ffill() over it. Because the origin is a real
        bbox rather than NaN, pandas never saw a gap: it interpolated *toward
        the origin* and back out, then guaranteed a ball on every frame --
        inventing one before the first detection and holding one after the
        last. Every frame got a confident answer and none of them were marked
        as estimates.

        Here, every returned frame is either a ball with an explicit `state`,
        or empty. Empty means unknown, and unknown is a real answer.

        Args:
            ball_positions: tracks['ball'], one dict per frame.
            candidates: per-frame candidate lists from get_object_tracks().
                None (e.g. tracks restored from an older cache) means no
                low-confidence evidence is available; accepted observations are
                still used, so the selector degrades to interpolation-only
                rather than silently rescuing nothing it should have.
            fps: effective frame rate of the frames actually processed, so a
                skipped-frame run scales its gap limits correctly.
            frame_width: source width in px; the gate is defined at 640.
            cuts: optional per-frame camera-cut flags from detect_cuts().

        Returns:
            List of ball dictionaries, one per frame, carrying 'state' and
            'confidence' (None for interpolated points, which are estimates
            and have no detector confidence).
        """
        from trackers.ball_temporal import (REFERENCE_WIDTH, UNKNOWN,
                                            BallTemporalSelector, FrameInput)

        n = len(ball_positions)
        rate = float(fps) if fps else self.frame_rate
        dt = 1.0 / rate if rate > 0 else 0.2

        inputs = []
        for i in range(n):
            if candidates is not None and i < len(candidates):
                cands = [dict(c) for c in candidates[i]]
            else:
                # No candidate record: fall back to whatever was accepted as an
                # observation, so the selector still anchors and interpolates.
                cands = [{'bbox': list(info['bbox']),
                          'confidence': float(info.get('confidence', 0.5))}
                         for info in ball_positions[i].values()
                         if info.get('bbox')]
            inputs.append(FrameInput(
                candidates=cands,
                timestamp=i * dt,
                dt=dt,
                cut=bool(cuts[i]) if cuts is not None and i < len(cuts) else False,
            ))

        selector = BallTemporalSelector(
            frame_width=float(frame_width) if frame_width else REFERENCE_WIDTH)
        results = selector.run(inputs)

        out = []
        for i, r in enumerate(results):
            if r.state == UNKNOWN or r.bbox is None:
                out.append({})          # honest gap; downstream must tolerate it
                continue
            ball_id = next(iter(ball_positions[i]), 1) if ball_positions[i] else 1
            entry = {'bbox': list(r.bbox), 'state': r.state}
            if r.confidence is not None:
                entry['confidence'] = r.confidence
            out.append({ball_id: entry})
        return out


    def detect_objects_in_frames(self, frames):
        """
        Detect objects (players, ball, referees) in video frames
        
        Args:
            frames: List of video frames
            
        Returns:
            List of detections per frame
        """
        all_detections = []
        batch_size = 10  # Process in smaller batches to avoid memory issues
        
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i+batch_size]
            batch_detections = []
            
            for frame_idx, frame in enumerate(batch_frames):
                # Get absolute frame index
                abs_idx = i + frame_idx
                detections = self.detector.detect(frame, abs_idx)
                batch_detections.append(detections)
                
                if abs_idx % 10 == 0:
                    print(f"Detected objects in frame {abs_idx}, found {len(detections)} objects")
            
            all_detections.extend(batch_detections)
            
        return all_detections
    
    def get_object_tracks(self, frames, read_from_cache=True, cache_path=None):
        """
        Get tracked objects from video frames
        
        Args:
            frames: List of video frames
            read_from_cache: Whether to read from cache if available
            cache_path: Path to cache file
            
        Returns:
            Dictionary of tracked objects
        """
        # Check cache first
        if read_from_cache and cache_path and os.path.exists(cache_path):
            print(f"Loading tracks from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                self.tracks = pickle.load(f)
            return self.tracks
        
        # Initialize tracks structure -- one list per detector class
        tracks = {key: [] for key in TRACK_KEYS}

        # Get detections for all frames
        detections_list = self.detect_objects_in_frames(frames)

        frames_since_ball = 0
        ball_candidates = []

        # Process each frame
        for frame_idx, frame_detections in enumerate(detections_list):
            # Prepare detection objects for supervision ByteTrack
            boxes = []
            class_ids = []
            confidences = []

            for det in frame_detections:
                class_name = det.get('class')
                if class_name not in CLASS_IDS:
                    continue  # detector already normalised; anything else is noise
                if class_name not in HUMAN_CLASSES:
                    # The ball never enters human association. It is not a
                    # person, it moves an order of magnitude faster, and letting
                    # it compete for IoU matches against players both wastes
                    # tracker state and produced spurious ball track ids -- 9 of
                    # them in a 12-second measured sequence. The ball has one
                    # canonical path, below.
                    continue
                # Humans below the accepted threshold DO enter the tracker when
                # the pool is on: they are exactly the low-score detections
                # ByteTrack's second association stage exists to consume. They
                # are withheld from the output below, not here. With the pool
                # off no such detection exists and this is a no-op.
                boxes.append(det['bbox'])
                class_ids.append(CLASS_IDS[class_name])
                confidences.append(det.get('confidence', 0.5))

            # Start this frame's slot for every class
            for key in TRACK_KEYS:
                tracks[key].append({})

            if boxes:
                detections = sv.Detections(
                    xyxy=np.array(boxes),
                    class_id=np.array(class_ids),
                    confidence=np.array(confidences)
                )

                # Update tracker. Both backends return a Detections carrying the
                # original class_id and confidence, so the identity is attached
                # to EyeCU's detection rather than replacing it: the semantic
                # class and the score are the detector's and are never rewritten
                # by association.
                if self.tracker_backend == 'cbiou':
                    tracked_detections = self.tracker.update(detections)
                else:
                    tracked_detections = self.tracker.update_with_detections(detections)

                for i in range(len(tracked_detections.xyxy)):
                    key = HUMAN_TRACK_KEY.get(int(tracked_detections.class_id[i]))
                    if key is None:
                        continue  # nothing but humans should reach here
                    raw_id = tracked_detections.tracker_id[i]
                    if raw_id is None or int(raw_id) < 0:
                        # Modern trackers report -1 for a track that has not yet
                        # met minimum_consecutive_frames. That is not an
                        # identity and must not become a dictionary key.
                        continue
                    raw_id = int(raw_id)
                    if self.tracker_backend != 'legacy':
                        # Modern trackers number from 0 (trackers 2.6.0,
                        # base_tracklet.py: tracker_id starts at -1 and the
                        # first confirmed track takes 0). EyeCU's public
                        # identity contract is POSITIVE integers, so the
                        # boundary shifts by one. Legacy sv.ByteTrack is
                        # already 1-based and is left alone. The map is
                        # injective, so uniqueness and continuity are preserved.
                        raw_id += 1
                    conf = float(tracked_detections.confidence[i])
                    if conf < self.human_accept_conf:
                        # Association evidence only. It kept the identity alive
                        # through this frame; it must not appear in reports,
                        # counts, statistics or the rendered video.
                        continue
                    track_id = raw_id
                    tracks[key][frame_idx][track_id] = {
                        "bbox": tracked_detections.xyxy[i].tolist(),
                        "confidence": conf,
                    }

            # The ball's single canonical path. It is written here and nowhere
            # else: ByteTrack no longer sees it, so it cannot also arrive above
            # under a tracker id. ID 1 keeps it stable for downstream code.
            ball_found = False
            candidates = []
            for det in frame_detections:
                if det.get('class') != 'ball':
                    continue
                # Every ball detection, at any confidence, is recorded as a
                # candidate. BallTemporalSelector adjudicates them downstream:
                # it is the only component allowed to promote a low-confidence
                # detection into a reported ball, and only inside a
                # motion-predicted gate. Collecting them here does not accept
                # them -- the observation test below is unchanged.
                candidates.append({
                    'bbox': list(det['bbox']),
                    'confidence': float(det.get('confidence', 0.5)),
                    'state': det.get('state', 'observed'),
                })
                # Detections carrying state='candidate_low_conf' are the rescue
                # pool and must never be accepted as observations here.
                # Detectors without the pool emit no 'state' key at all, and
                # the default keeps their behaviour identical.
                if det.get('state', 'observed') != 'observed':
                    continue
                if not ball_found:
                    tracks["ball"][frame_idx][1] = {
                        "bbox": det['bbox'],
                        "confidence": det.get('confidence', 0.5),
                    }
                    ball_found = True

            # Kept per frame and aligned with tracks['ball'] by index, so the
            # selector can be applied after tracking without re-running the
            # detector. Empty list means "no ball candidate at any confidence".
            ball_candidates.append(candidates)

            if ball_found:
                frames_since_ball = 0
            else:
                frames_since_ball += 1
                # Hold the last known box for a bounded number of frames only.
                # Copying it indefinitely used to freeze a stale ball on screen
                # for the rest of the video and corrupt possession statistics.
                if 0 < frames_since_ball <= self.max_ball_gap and frame_idx > 0:
                    for ball_id, ball_info in tracks["ball"][frame_idx - 1].items():
                        held = ball_info.copy()
                        held['held_for'] = frames_since_ball
                        tracks["ball"][frame_idx][ball_id] = held
                # Beyond max_ball_gap the ball is simply unknown: leave it empty.

        # Save to cache if requested
        if self.persist_cache and cache_path:
            print(f"Saving tracks to cache: {cache_path}")
            with open(cache_path, 'wb') as f:
                pickle.dump(tracks, f)

        # Cached so generate_final_report() can write player_statistics.json.
        self.tracks = tracks
        # Read by apply_ball_temporal_selection(). Not part of tracks[] because
        # it is detector evidence, not a track, and must not reach consumers
        # that iterate tracks by class.
        self.ball_candidates = ball_candidates
        return tracks
    
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        """
        Draw ellipse under player/referee feet
        
        Args:
            frame: Frame to draw on
            bbox: Bounding box [x1, y1, x2, y2]
            color: Color tuple (B, G, R)
            track_id: Optional tracking ID to display
            
        Returns:
            Annotated frame
        """
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        # Draw a more prominent ellipse
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=3,  # Increased thickness
            lineType=cv2.LINE_4
        )

        if track_id is not None:
            rectangle_width = 40
            rectangle_height = 20
            x1_rect = x_center - rectangle_width//2
            x2_rect = x_center + rectangle_width//2
            y1_rect = (y2- rectangle_height//2) + 15
            y2_rect = (y2 + rectangle_height//2) + 15

            # Draw a bigger ID box
            cv2.rectangle(frame,
                        (int(x1_rect), int(y1_rect)),
                        (int(x2_rect), int(y2_rect)),
                        color,
                        cv2.FILLED)
                        
            # Add a black outline to make it more visible
            cv2.rectangle(frame,
                        (int(x1_rect), int(y1_rect)),
                        (int(x2_rect), int(y2_rect)),
                        (0, 0, 0),
                        1)
            
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10
                
            # Draw a white background for the player ID text for better visibility
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,  # Slightly larger font
                (255, 255, 255),  # White outline
                3     # Thicker outline
            )
            
            # Draw the player ID in a contrasting color on top
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,  # Slightly larger font
                (0, 0, 0),  # Black text
                1     # Regular thickness
            )

        return frame
    
    def draw_triangle(self, frame, bbox, color):
        """
        Draw triangle for ball
        
        Args:
            frame: Frame to draw on
            bbox: Bounding box [x1, y1, x2, y2]
            color: Color tuple (B, G, R)
            
        Returns:
            Annotated frame
        """
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        # Make the triangle larger and more visible
        triangle_points = np.array([
            [x, y],
            [x-15, y-25],  # Increased size
            [x+15, y-25],  # Increased size
        ], np.int32)
        
        # Draw a filled triangle with the color
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        
        # Draw a slightly thicker black outline
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame
    
    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        """
        Draw team ball control statistics
        
        Args:
            frame: Frame to draw on
            frame_num: Current frame number
            team_ball_control: Array of team IDs controlling ball
            
        Returns:
            Annotated frame
        """
        # Draw a semi-transparent rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        # Get the number of times each team had ball control
        team_1_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        total_frames = team_1_frames + team_2_frames
        
        if total_frames > 0:
            team_1_pct = team_1_frames / total_frames
            team_2_pct = team_2_frames / total_frames
        else:
            team_1_pct = team_2_pct = 0.0

        cv2.putText(frame, f"Team 1 Ball Control: {team_1_pct*100:.2f}%",
                  (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2_pct*100:.2f}%",
                  (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        return frame
    
    def draw_annotations(self, video_frames, tracks, team_ball_control=None):
        """
        Draw all annotations on video frames
        
        Args:
            video_frames: List of video frames
            tracks: Dictionary of tracked objects
            team_ball_control: Array of team IDs controlling ball
            
        Returns:
            List of annotated frames
        """
        output_frames = []
        
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            
            # Add detection counts at the top
            player_count = len(tracks["players"][frame_num])
            keeper_count = len(tracks["goalkeepers"][frame_num])
            referee_count = len(tracks["referees"][frame_num])
            ball_count = len(tracks["ball"][frame_num])

            # Draw info bar
            cv2.rectangle(frame, (0, 0), (400, 105), (0, 0, 0), -1)
            cv2.putText(frame, f"Players: {player_count}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Goalkeepers: {keeper_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, self.colors['goalkeeper'], 2)
            cv2.putText(frame, f"Referees: {referee_count}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 0), 2)
            cv2.putText(frame, f"Ball: {ball_count}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)

            # Draw players
            player_dict = tracks["players"][frame_num]
            for track_id, player in player_dict.items():
                # Get team color if available
                color = player.get("team_color", self.colors['player'])
                # Draw a more prominent bounding box
                bbox = player["bbox"]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)
                
                # Highlight player with ball
                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player["bbox"], (0, 0, 255))
            
            # Draw goalkeepers -- kept visually distinct from outfield players
            # because their kit deliberately differs from their own team's.
            for track_id, keeper in tracks["goalkeepers"][frame_num].items():
                bbox = keeper["bbox"]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                              self.colors['goalkeeper'], 2)
                frame = self.draw_ellipse(frame, bbox, self.colors['goalkeeper'], track_id)
                if keeper.get('has_ball', False):
                    frame = self.draw_triangle(frame, bbox, (0, 0, 255))

            # Draw referees
            referee_dict = tracks["referees"][frame_num]
            for track_id, referee in referee_dict.items():
                bbox = referee["bbox"]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), 
                              self.colors['referee'], 2)
                frame = self.draw_ellipse(frame, referee["bbox"], self.colors['referee'], track_id)
            
            # Draw ball
            ball_dict = tracks["ball"][frame_num]
            for track_id, ball in ball_dict.items():
                bbox = ball["bbox"]
                # Draw a circle for ball for better visibility
                center = get_center_of_bbox(bbox)
                radius = max(5, int((bbox[2] - bbox[0]) / 2))
                cv2.circle(frame, (center[0], center[1]), radius, self.colors['ball'], -1)
                frame = self.draw_triangle(frame, ball["bbox"], self.colors['ball'])
            
            # Draw team ball control stats if available
            if team_ball_control is not None:
                frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)
            
            output_frames.append(frame)
        
        return output_frames
    
    # Utility functions
    # Helper functions are now imported from bbox_utils.py
