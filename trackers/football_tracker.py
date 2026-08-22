"""
Football player tracking.

Association runs on a selectable backend -- CBIoU by default, supervision
ByteTrack under tracker_backend='legacy'. Humans are tracked; the ball takes a
separate path and is resolved by BallTemporalSelector, which tags every
reported position with its provenance.

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
from trackers import overlay as ui

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
# Only these reach the association backend. The ball is excluded by
# construction rather than by a downstream filter, so a future edit cannot
# silently readmit it.
HUMAN_TRACK_KEY = {CLASS_IDS[name]: TRACK_KEY[name] for name in HUMAN_CLASSES}


class FootballTracker:
    """
    Comprehensive football video analysis tracker.

    Tracks players, goalkeepers, referees and the ball as distinct classes.
    Human association uses the configured backend (CBIoU by default); the ball
    bypasses it entirely and is resolved temporally instead.
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
                ball_detector_backend='eyecu',
                ball_model_path=None,
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
            ball_detector_backend: which model supplies the BALL class.
                'eyecu' -- the single EyeCU detector supplies all four classes,
                           the behaviour in force before D2 and still the
                           DEFAULT.
                'sn3d'  -- humans stay on the EyeCU detector, unchanged, and the
                           ball comes from the official SoccerNet-v3D
                           yolo-sn-ball.pt at its trained 1280, selected by the
                           S1B/S1C/S1D gates. Requires ball_model_path.
                Either way the ball never enters human association.
            ball_model_path: path to yolo-sn-ball.pt. Required by, and only used
                by, ball_detector_backend='sn3d'. Its SHA256 is verified against
                the pinned SN3D_BALL_SHA256 before it runs.
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

        self.ball_detector_backend = ball_detector_backend
        self.ball_model_path = ball_model_path
        self.detector = detector if detector is not None else create_detector(
            model_path=model_path,
            use_roboflow=use_roboflow,
            api_key=api_key,
            confidence=confidence,
            imgsz=imgsz,
            ball_candidate_pool=ball_candidate_pool,
            human_candidate_pool=human_candidate_pool,
            ball_detector_backend=ball_detector_backend,
            ball_model_path=ball_model_path,
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
        # Per-frame DETECTOR candidates, aligned with tracks['ball'] by index.
        # None means no run has produced them yet. It is never a licence to
        # substitute tracks['ball']: that holds tracker output, including boxes
        # copied forward during a gap.
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

        Three things are deliberately kept apart here:

            A  detector candidate   -- a raw detection at any confidence
            B  tracker observation  -- what tracks['ball'] holds, which
                                       INCLUDES boxes copied forward by the
                                       bounded hold
            C  selector output      -- what this method returns

        Only A may enter the selector as evidence. An earlier version fell back
        to B when candidates were missing, which relabelled a held box as
        `observed` carrying the original detector's confidence -- a copied
        position presented as a measurement. That fallback is gone.

        Args:
            ball_positions: tracks['ball'], one dict per frame. Used ONLY for
                the frame count and to keep the ball's track id stable. Its
                bboxes are never read as evidence.
            candidates: per-frame detector candidate lists (A) from
                get_object_tracks(). Required. Passing None raises rather than
                guessing, because every silent alternative fabricates evidence.
            fps: effective frame rate of the frames actually processed, so a
                skipped-frame run scales its gap limits correctly.
            frame_width: source width in px; the gate is defined at 640.
            cuts: optional per-frame camera-cut flags from detect_cuts().

        Returns:
            List of ball dictionaries, one per frame, carrying 'state' and
            'confidence' (None for interpolated points, which are estimates
            and have no detector confidence).

        Raises:
            ValueError: if `candidates` is None or shorter than the frames.
        """
        from trackers.ball_temporal import (REFERENCE_WIDTH, UNKNOWN,
                                            BallTemporalSelector, FrameInput)

        n = len(ball_positions)
        if candidates is None:
            raise ValueError(
                'apply_ball_temporal_selection() requires detector candidates. '
                'They are produced by get_object_tracks() and stored in the v2 '
                'cache. Reconstructing them from tracks["ball"] is not a valid '
                'substitute: that list contains boxes held forward by the '
                'tracker, and feeding them back would present a copied '
                'position as a measurement.')
        if len(candidates) < n:
            raise ValueError(
                f'candidate record covers {len(candidates)} frames but there '
                f'are {n} ball frames; refusing to guess the remainder')

        rate = float(fps) if fps else self.frame_rate
        dt = 1.0 / rate if rate > 0 else 0.2

        inputs = []
        for i in range(n):
            inputs.append(FrameInput(
                candidates=[dict(c) for c in candidates[i]],
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
    
    # Bumped when the cached payload gains a field the pipeline depends on.
    # v1 was the bare tracks dict; v2 adds the detector candidate pool, which
    # BallTemporalSelector needs and cannot be reconstructed from tracks.
    CACHE_FORMAT = 2

    @staticmethod
    def _unpack_cache(blob):
        """(tracks, ball_candidates) from a cache payload, or (None, None).

        A v1 cache is REJECTED rather than upgraded. The tempting upgrade --
        deriving candidates from tracks['ball'] -- is exactly the bug this
        replaces: that list holds *tracker* boxes, including ones copied
        forward by the bounded hold, and feeding them back as detector
        evidence relabels a held box as a measurement. Recomputing costs one
        detection pass; the alternative silently corrupts provenance.
        """
        if isinstance(blob, dict) and blob.get('cache_format') == \
                FootballTracker.CACHE_FORMAT:
            return blob.get('tracks'), blob.get('ball_candidates')
        return None, None

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
            with open(cache_path, 'rb') as f:
                blob = pickle.load(f)
            tracks, candidates = self._unpack_cache(blob)
            if tracks is not None:
                print(f"Loading tracks from cache: {cache_path}")
                self.tracks = tracks
                self.ball_candidates = candidates
                return self.tracks
            # v1 cache: tracks only, no detector candidates. Recomputed rather
            # than reused, because the alternative is running the temporal
            # selector on evidence it was never given -- see _unpack_cache().
            print(f"Ignoring stale cache without ball candidates: {cache_path}")


        # Initialize tracks structure -- one list per detector class
        tracks = {key: [] for key in TRACK_KEYS}

        # Get detections for all frames
        detections_list = self.detect_objects_in_frames(frames)

        frames_since_ball = 0
        ball_candidates = []

        # Process each frame
        for frame_idx, frame_detections in enumerate(detections_list):
            # Prepare detection objects for the association backend
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
            # else: the association backend no longer sees it, so it cannot
            # also arrive above
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
                # Candidates travel WITH the tracks. Saving tracks alone made a
                # cached run and a fresh run disagree about the ball, because
                # only the fresh one still had the detector's candidate pool.
                pickle.dump({'cache_format': self.CACHE_FORMAT,
                             'tracks': tracks,
                             'ball_candidates': ball_candidates}, f)

        # Cached so generate_final_report() can write player_statistics.json.
        self.tracks = tracks
        # Read by apply_ball_temporal_selection(). Not part of tracks[] because
        # it is detector evidence, not a track, and must not reach consumers
        # that iterate tracks by class.
        self.ball_candidates = ball_candidates
        return tracks
    
    def draw_ellipse(self, frame, bbox, color, track_id=None, style=None):
        """
        Draw ellipse under player/referee feet (DEBUG-mode marker style).

        Args:
            frame: Frame to draw on
            bbox: Bounding box [x1, y1, x2, y2]
            color: Color tuple (B, G, R)
            track_id: Optional tracking ID to display
            style: Optional ui.get_ui_scale(...) dict; a frame-height-720
                default is used if omitted (keeps old callers working).

        Returns:
            Annotated frame
        """
        if style is None:
            style = ui.get_ui_scale(frame.shape[1], frame.shape[0])
        scale = style['scale']

        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=max(1, round(3 * scale)),
            lineType=cv2.LINE_4
        )

        if track_id is not None:
            rectangle_width = max(24, round(40 * scale))
            rectangle_height = max(12, round(20 * scale))
            x1_rect = x_center - rectangle_width//2
            x2_rect = x_center + rectangle_width//2
            y1_rect = (y2 - rectangle_height//2) + round(15 * scale)
            y2_rect = (y2 + rectangle_height//2) + round(15 * scale)

            cv2.rectangle(frame,
                        (int(x1_rect), int(y1_rect)),
                        (int(x2_rect), int(y2_rect)),
                        color,
                        cv2.FILLED)
            cv2.rectangle(frame,
                        (int(x1_rect), int(y1_rect)),
                        (int(x2_rect), int(y2_rect)),
                        (0, 0, 0),
                        1)

            font_scale = round(0.7 * scale, 3)
            (tw, th), _ = cv2.getTextSize(f"{track_id}", cv2.FONT_HERSHEY_SIMPLEX,
                                          font_scale, max(1, round(2 * scale)))
            text_x = int(x_center - tw / 2)
            text_y = int(y1_rect + rectangle_height/2 + th/2)
            ui.draw_text_with_outline(frame, f"{track_id}", (text_x, text_y),
                                       font_scale, (0, 0, 0), max(1, round(1 * scale)),
                                       outline_color=(255, 255, 255))

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
    
    def draw_team_ball_control(self, frame, frame_num, team_ball_control, style=None):
        """
        Draw team ball control statistics (DEBUG mode).

        Args:
            frame: Frame to draw on
            frame_num: Current frame number
            team_ball_control: Array of team IDs controlling ball
            style: Optional ui.get_ui_scale(...) dict.

        Returns:
            Annotated frame
        """
        if style is None:
            style = ui.get_ui_scale(frame.shape[1], frame.shape[0])
        fw, fh = frame.shape[1], frame.shape[0]
        panel_w = max(180, round(260 * style['scale']))
        panel_h = max(70, round(85 * style['scale']))
        margin = style['margin']
        x1 = fw - panel_w - margin
        y1 = fh - panel_h - margin
        x2 = fw - margin
        y2 = fh - margin

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_1_pct, team_2_pct, _ = self._team_possession_pct(team_ball_control, frame_num)

        font_scale = round(1.0 * style['scale'], 3)
        thick = max(1, round(3 * style['scale']))
        line_h = max(16, round(28 * style['scale']))
        tx = x1 + max(8, round(12 * style['scale']))
        ty1 = y1 + round(line_h * 1.1)
        ty2 = ty1 + line_h
        cv2.putText(frame, f"Team 1 Ball Control: {team_1_pct*100:.2f}%",
                  (tx, ty1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thick)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2_pct*100:.2f}%",
                  (tx, ty2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thick)

        return frame
    
    @staticmethod
    def _team_possession_pct(team_ball_control, frame_num):
        """
        Known-possession-frames-only percentage split, shared by the DEBUG
        panel and the VIEWER HUD. 0 (unknown) is excluded from the
        denominator -- see the comment this replaced in
        draw_team_ball_control for why that matters.
        """
        till_frame = team_ball_control[:frame_num + 1]
        team_1_frames = till_frame[till_frame == 1].shape[0]
        team_2_frames = till_frame[till_frame == 2].shape[0]
        total = team_1_frames + team_2_frames
        if total > 0:
            return team_1_frames / total, team_2_frames / total, total
        return 0.0, 0.0, 0

    def draw_annotations(self, video_frames, tracks, team_ball_control=None,
                          overlay_mode='viewer'):
        """
        Draw all annotations on video frames.

        This is presentation only: overlay_mode picks how the same tracks[]
        are drawn ('viewer' = clean tactical-camera style for demo/final
        video, 'debug' = the original engineering overlay, made responsive).
        It never changes what was detected, tracked, selected or assigned.

        Args:
            video_frames: List of video frames
            tracks: Dictionary of tracked objects
            team_ball_control: Array of team IDs controlling ball
            overlay_mode: 'viewer' (default) or 'debug'

        Returns:
            List of annotated frames
        """
        if overlay_mode not in ('viewer', 'debug'):
            raise ValueError(f"unknown overlay_mode {overlay_mode!r}")
        if overlay_mode == 'viewer':
            return self._draw_viewer_annotations(video_frames, tracks, team_ball_control)
        return self._draw_debug_annotations(video_frames, tracks, team_ball_control)

    def _draw_debug_annotations(self, video_frames, tracks, team_ball_control=None):
        """Original engineering overlay: full box + ellipse + ID + counts + panel."""
        output_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            style = ui.get_ui_scale(frame.shape[1], frame.shape[0])
            scale = style['scale']

            player_count = len(tracks["players"][frame_num])
            keeper_count = len(tracks["goalkeepers"][frame_num])
            referee_count = len(tracks["referees"][frame_num])
            ball_count = len(tracks["ball"][frame_num])

            bar_w = max(180, round(300 * scale))
            bar_h = max(60, round(90 * scale))
            font_scale = round(0.7 * scale, 3)
            thick = max(1, round(2 * scale))
            line_h = max(14, round(24 * scale))
            cv2.rectangle(frame, (0, 0), (bar_w, bar_h), (0, 0, 0), -1)
            cv2.putText(frame, f"Players: {player_count}", (10, line_h), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (0, 255, 0), thick)
            cv2.putText(frame, f"Goalkeepers: {keeper_count}", (10, line_h * 2), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, self.colors['goalkeeper'], thick)
            cv2.putText(frame, f"Referees: {referee_count}", (10, line_h * 3), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (255, 255, 0), thick)
            cv2.putText(frame, f"Ball: {ball_count}", (10, line_h * 4), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (0, 0, 255), thick)

            box_thick = max(1, round(2 * scale))

            player_dict = tracks["players"][frame_num]
            for track_id, player in player_dict.items():
                color = player.get("team_color", self.colors['player'])
                bbox = player["bbox"]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, box_thick)
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id, style=style)
                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player["bbox"], (0, 0, 255))

            for track_id, keeper in tracks["goalkeepers"][frame_num].items():
                bbox = keeper["bbox"]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                              self.colors['goalkeeper'], box_thick)
                frame = self.draw_ellipse(frame, bbox, self.colors['goalkeeper'], track_id, style=style)
                if keeper.get('has_ball', False):
                    frame = self.draw_triangle(frame, bbox, (0, 0, 255))

            referee_dict = tracks["referees"][frame_num]
            for track_id, referee in referee_dict.items():
                bbox = referee["bbox"]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                              self.colors['referee'], box_thick)
                frame = self.draw_ellipse(frame, referee["bbox"], self.colors['referee'], track_id, style=style)

            ball_dict = tracks["ball"][frame_num]
            for track_id, ball in ball_dict.items():
                bbox = ball["bbox"]
                center = get_center_of_bbox(bbox)
                radius = max(5, int((bbox[2] - bbox[0]) / 2))
                cv2.circle(frame, (center[0], center[1]), radius, self.colors['ball'], -1)
                frame = self.draw_triangle(frame, ball["bbox"], self.colors['ball'])

            if team_ball_control is not None:
                frame = self.draw_team_ball_control(frame, frame_num, team_ball_control, style=style)

            output_frames.append(frame)

        return output_frames

    def _draw_viewer_annotations(self, video_frames, tracks, team_ball_control=None):
        """
        Clean, tactical-camera-friendly overlay for demo/final rendering.
        Small feet markers + compact ID pills (role-differentiated by style,
        not color alone), a possessor halo when the pipeline has already
        assigned one, a light-ring ball marker keyed to selector state, and a
        compact top-left HUD. No detection/tracking/selection/possession
        value is computed here -- only how it is drawn.
        """
        output_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            style = ui.get_ui_scale(frame.shape[1], frame.shape[0])
            occupied = []

            entries = []
            for track_id, gk in tracks["goalkeepers"][frame_num].items():
                entries.append((ui.ROLE_PRIORITY['goalkeeper'], track_id, gk, 'goalkeeper'))
            for track_id, pl in tracks["players"][frame_num].items():
                role_priority = ui.ROLE_PRIORITY['possessor'] if pl.get('has_ball', False) \
                    else ui.ROLE_PRIORITY['player']
                entries.append((role_priority, track_id, pl, 'player'))
            for track_id, ref in tracks["referees"][frame_num].items():
                entries.append((ui.ROLE_PRIORITY['referee'], track_id, ref, 'referee'))
            entries.sort(key=lambda e: e[0])

            for _, track_id, obj, role in entries:
                bbox = obj["bbox"]
                foot = get_foot_position(bbox)
                if role == 'referee':
                    fill_color = (140, 140, 140)  # neutral steel-grey, not a team color
                elif role == 'goalkeeper':
                    fill_color = self.colors['goalkeeper']
                else:
                    fill_color = obj.get("team_color", self.colors['player'])
                is_possessor = obj.get('has_ball', False)
                ui.draw_player_marker(frame, foot, role, fill_color, style,
                                       track_id=track_id, is_possessor=is_possessor,
                                       occupied=occupied)

            ball_dict = tracks["ball"][frame_num]
            for _, ball in ball_dict.items():
                center = get_center_of_bbox(ball["bbox"])
                state = ball.get('state', 'observed')
                ui.draw_ball_marker(frame, center, state, style)

            # VIEWER HUD is deliberately minimal: no team-possession percentages
            # (possession remains CLOSED-LIMITATION -- not a display-worthy
            # accuracy claim) and no processed-frame index (an engineering
            # detail, not viewer-facing information). Just who has the ball,
            # if anyone -- and that lookup includes ANY eligible human (not
            # only role == 'player'), so a goalkeeper possession is shown too.
            hud_lines = []
            possessor_id = next((tid for _, tid, obj, role in entries
                                 if role in ('player', 'goalkeeper') and obj.get('has_ball', False)), None)
            if possessor_id is not None:
                hud_lines.append(f"Possessor #{possessor_id}")
            ui.draw_viewer_hud(frame, style, hud_lines[:5])

            output_frames.append(frame)

        return output_frames

    # Utility functions
    # Helper functions are now imported from bbox_utils.py
