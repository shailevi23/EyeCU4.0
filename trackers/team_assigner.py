"""
Team Assignment based on Jersey Colors
Uses KMeans clustering to assign players to teams based on jersey colors
"""

import numpy as np
from sklearn.cluster import KMeans
from typing import Dict, List


class TeamAssigner:
    """
    Assign players to teams based on jersey colors using KMeans clustering
    """
    
    def __init__(self, num_teams: int = 2, 
                use_top_half: bool = True):
        """
        Initialize team assigner
        
        Args:
            num_teams: Number of teams to detect
            use_top_half: Whether to use only top half of player for color detection
        """
        self.num_teams = num_teams
        self.use_top_half = use_top_half
        self.team_colors = {}  # Detected jersey colors (BGR), used only for clustering
        # Fixed, mutually distinct colors for on-screen annotation (BGR) - the real jersey
        # color is often low-contrast against itself (e.g. white-on-white, yellow-on-yellow),
        # so drawing overlays never reuse it.
        self.display_colors = {1: (255, 60, 60), 2: (60, 60, 255)}  # blue, red
        self.player_team_dict = {}  # Maps player_id to team_id
        self.kmeans = None
    

    def get_player_color(self, frame: np.ndarray, 
                        bbox: List[float]) -> np.ndarray:
        """
        Get dominant jersey color for a player
        
        Args:
            frame: Video frame
            bbox: Player bounding box [x1, y1, x2, y2]
            
        Returns:
            RGB color value as numpy array
        """
        # Extract player crop from frame
        x1, y1, x2, y2 = map(int, bbox)
        player_img = frame[y1:y2, x1:x2]
        
        if player_img.size == 0:
            # Return default color if crop is empty
            return np.array([128, 128, 128])
        
        # Sample a central chest ROI instead of clustering the whole crop: on tight
        # bounding boxes the old corner-based background heuristic often picked the
        # wrong cluster (arms/shorts/grass), making different jerseys average to the
        # same color. The chest region is almost always pure jersey fabric.
        h, w = player_img.shape[:2]
        y_start, y_end = int(h * 0.15), int(h * 0.50)
        x_start, x_end = int(w * 0.25), int(w * 0.75)
        torso = player_img[y_start:y_end, x_start:x_end]
        
        if torso.size == 0:
            torso = player_img
        
        return np.median(torso.reshape(-1, 3), axis=0)
    
    @staticmethod
    def _bbox_overlap_ratio(box_a: List[float], box_b: List[float]) -> float:
        """Fraction of box_a's own area covered by box_b - directional, so a player
        whose box is mostly swallowed by a neighbor's box is flagged even if the
        neighbor (larger) box isn't."""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(1e-6, (ax2 - ax1) * (ay2 - ay1))
        return inter / area_a
    
    def _collect_color_samples(self, frames: List[np.ndarray], tracks: Dict,
                              frame_indices: range, skip_overlaps: bool) -> Dict[int, List[np.ndarray]]:
        """Sample each player's chest color per frame. When skip_overlaps is True,
        frames where a player's box is heavily covered by a neighbor's box are
        excluded - those crops are usually contaminated with the neighbor's jersey
        color, which corrupts both the team-color fit and any per-player average."""
        samples: Dict[int, List[np.ndarray]] = {}
        for frame_idx in frame_indices:
            if frame_idx >= len(tracks['players']):
                continue
            frame = frames[frame_idx]
            players = tracks['players'][frame_idx]
            for player_id, player_data in players.items():
                bbox = player_data.get('bbox', [0, 0, 0, 0])
                if skip_overlaps:
                    overlapped = any(
                        self._bbox_overlap_ratio(bbox, other_data.get('bbox', [0, 0, 0, 0])) > 0.25
                        for other_id, other_data in players.items() if other_id != player_id
                    )
                    if overlapped:
                        continue
                samples.setdefault(player_id, []).append(self.get_player_color(frame, bbox))
        return samples
    
    def _fit_team_colors(self, player_colors: np.ndarray) -> None:
        """
        Fit the team-color model on a pool of sampled player colors
        
        Args:
            player_colors: Nx3 array of BGR player colors
        """
        # Referees/goalkeepers often get mislabeled as generic "player" detections and end
        # up mixed into this sample pool. Officials in this footage wear a solid dark/black
        # kit, which is much darker than either real team's (bright) jersey, so we filter
        # out very dark samples before fitting - this is more reliable than trying to
        # separate outliers via over-clustering, since a naive nearest-pair merge can just
        # as easily merge the two real (both bright) team colors together and isolate the
        # outlier as its own "team" instead.
        brightness = player_colors.max(axis=1)
        is_dark_outlier = brightness < 100
        fit_colors = player_colors[~is_dark_outlier]
        if len(fit_colors) < self.num_teams:
            # Not enough bright samples to trust the filter - fall back to using everything
            fit_colors = player_colors
        
        kmeans = KMeans(n_clusters=self.num_teams, init="k-means++", n_init=10)
        kmeans.fit(fit_colors)
        self.kmeans = kmeans
        
        # Store team colors
        for i in range(self.num_teams):
            # frame/player_color are already BGR (frames are never converted to RGB), so no channel reversal needed
            color_bgr = kmeans.cluster_centers_[i].astype(np.uint8)
            self.team_colors[i+1] = (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2]))
        
        print(f"Team colors assigned: {self.team_colors}")
    
    def get_player_team(self, frame: np.ndarray, 
                       player_bbox: List[float], 
                       player_id: int) -> int:
        """
        Get team assignment for a player
        
        Args:
            frame: Video frame
            player_bbox: Player bounding box
            player_id: Player tracking ID
            
        Returns:
            Team ID (1-based)
        """
        # Check if we already know this player's team
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        # Extract player color and predict team
        if self.kmeans is None:
            # If we haven't assigned team colors yet, assign to team 1
            team_id = 1
        else:
            player_color = self.get_player_color(frame, player_bbox)
            
            # Predict team based on color
            team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
            team_id += 1  # Convert to 1-based indexing
        
        # Special case handling for goalkeepers or specific players if needed
        # (commented out as it depends on specific use case)
        # if player_id == 91:  # Example: always assign player 91 to team 1
        #     team_id = 1
        
        # Store assignment for future reference
        self.player_team_dict[player_id] = team_id
        
        return team_id
    
    def assign_teams_to_tracks(self, frames: List[np.ndarray], 
                              tracks: Dict) -> None:
        """
        Assign teams to all players in tracking data
        
        Args:
            frames: List of video frames
            tracks: Dictionary of tracking data
        """
        # Fit team colors on samples pooled across many frames (not just one). A single
        # frame usually only has ~8-12 players, and goalkeepers (normalized to the
        # generic 'player' class upstream) wear a third, distinctly colored kit - with
        # so few samples their odd color can hijack a whole cluster and force both real
        # outfield teams into the other one. Pooling many frames dilutes that outlier.
        # Overlapping boxes (players standing next to/behind each other) are skipped
        # since their crops mix in a neighbor's jersey color.
        num_sample_frames = min(20, len(frames))
        fit_samples = self._collect_color_samples(frames, tracks, range(num_sample_frames), skip_overlaps=True)
        sample_colors = [c for colors in fit_samples.values() for c in colors]
        if len(sample_colors) < self.num_teams * 3:
            # Not enough non-overlapping samples to trust the filter - fall back to everything
            fit_samples = self._collect_color_samples(frames, tracks, range(num_sample_frames), skip_overlaps=False)
            sample_colors = [c for colors in fit_samples.values() for c in colors]
        
        if len(sample_colors) < self.num_teams:
            print(f"Warning: Found only {len(sample_colors)} player color samples, need at least {self.num_teams} for team assignment")
        else:
            self._fit_team_colors(np.array(sample_colors))
        
        # Decide each player's team from the MEDIAN color across every frame they appear
        # in, not just the first sighting. get_player_team() caches its result per
        # player_id forever, so if it were driven frame-by-frame a single occluded/blurry
        # first frame would permanently mislabel that track for the whole video.
        # Overlap-contaminated frames are excluded per player, falling back to every
        # frame only for players that are overlapped in ALL of their appearances.
        clean_samples = self._collect_color_samples(frames, tracks, range(len(frames)), skip_overlaps=True)
        all_samples = self._collect_color_samples(frames, tracks, range(len(frames)), skip_overlaps=False)
        
        for player_id, colors in all_samples.items():
            use_colors = clean_samples.get(player_id) or colors
            # Even non-overlapping crops can be transiently darkened by shadow, motion
            # blur, or occlusion by something other than a tracked player (the ball,
            # a limb, camera pan blur). Both real team jerseys here are bright, so a
            # genuine player's dark readings are noise, not signal - prefer the bright
            # subset when there is one. A track that's consistently dark in nearly all
            # its samples (the referee) has no bright subset, so it falls back to using
            # everything and correctly keeps its dark color.
            use_colors_arr = np.array(use_colors)
            bright_mask = use_colors_arr.max(axis=1) >= 150
            if bright_mask.sum() >= max(3, len(use_colors_arr) * 0.3):
                use_colors_arr = use_colors_arr[bright_mask]
            median_color = np.median(use_colors_arr, axis=0)
            if self.kmeans is None:
                team_id = 1
            else:
                team_id = int(self.kmeans.predict(median_color.reshape(1, -1))[0]) + 1
            self.player_team_dict[player_id] = team_id
        
        # Then, assign teams to all tracked players
        for frame_idx, frame in enumerate(frames):
            # Skip if no players in this frame
            if frame_idx >= len(tracks['players']):
                continue
                
            for player_id, player_data in tracks['players'][frame_idx].items():
                team_id = self.player_team_dict.get(player_id, 1)
                
                # Store team assignment and color in tracking data
                tracks['players'][frame_idx][player_id]['team'] = team_id
                tracks['players'][frame_idx][player_id]['team_color'] = self.display_colors.get(
                    team_id, (0, 255, 0))  # Default green if team color not found
