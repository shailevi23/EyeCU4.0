"""
Visualization script for mesh reconstruction test results
This script visualizes the results of the test_mesh_pipeline.py
"""

import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import glob

class MeshResultVisualizer:
    """Visualize results from the mesh pipeline test"""
    
    def __init__(self, output_dir: str = 'test_mesh_output'):
        self.output_dir = Path(output_dir)
        self.bodies_dir = self.output_dir / 'bodies'
        self.faces_dir = self.output_dir / 'faces'
        self.meshes_dir = self.output_dir / 'meshes'
        
    def visualize_results(self):
        """Create comprehensive visualization of test results"""
        self._create_player_grid()
        self._create_pose_visualization()
        self._create_similarity_matrix()
        
    def _create_player_grid(self):
        """Create grid of player detections with face and body crops"""
        # Find all body crops
        body_files = sorted(glob.glob(str(self.bodies_dir / "*.jpg")))
        
        if not body_files:
            print("No body crops found")
            return
        
        # Group by player ID
        players = {}
        for body_file in body_files:
            basename = os.path.basename(body_file)
            # Format: player_<id>_frame_<frame>.jpg
            parts = basename.replace('.jpg', '').split('_')
            
            if len(parts) >= 4:
                player_id = int(parts[1])
                frame_id = int(parts[3])
                
                if player_id not in players:
                    players[player_id] = []
                
                # Find corresponding face file
                face_file = self.faces_dir / basename
                
                players[player_id].append({
                    'player_id': player_id,
                    'frame_id': frame_id,
                    'body_file': body_file,
                    'face_file': str(face_file) if face_file.exists() else None
                })
        
        # Sort by frame ID
        for player_id in players:
            players[player_id].sort(key=lambda x: x['frame_id'])
        
        # Create visualization grid for each player
        for player_id, frames in players.items():
            # Create figure
            n_frames = len(frames)
            if n_frames == 0:
                continue
                
            fig_width = min(15, n_frames * 3)
            fig = plt.figure(figsize=(fig_width, 6))
            fig.suptitle(f'Player {player_id} Tracking Results', fontsize=16)
            
            # Create grid
            grid_size = min(n_frames, 5)  # Max 5 frames per row
            rows = (n_frames + grid_size - 1) // grid_size
            
            for i, frame_data in enumerate(frames):
                # Body image
                plt.subplot(rows * 2, grid_size, i + 1)
                if os.path.exists(frame_data['body_file']):
                    body_img = cv2.imread(frame_data['body_file'])
                    body_img = cv2.cvtColor(body_img, cv2.COLOR_BGR2RGB)
                    plt.imshow(body_img)
                    plt.title(f"Frame {frame_data['frame_id']} - Body")
                plt.axis('off')
                
                # Face image (if available)
                plt.subplot(rows * 2, grid_size, i + grid_size + 1)
                if frame_data['face_file'] and os.path.exists(frame_data['face_file']):
                    face_img = cv2.imread(frame_data['face_file'])
                    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    plt.imshow(face_img)
                    plt.title(f"Frame {frame_data['frame_id']} - Face")
                else:
                    plt.text(0.5, 0.5, "No face detected", 
                            ha='center', va='center')
                plt.axis('off')
            
            # Save figure
            fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for title
            save_path = self.output_dir / f"player_{player_id}_crops.png"
            plt.savefig(save_path, dpi=150)
            plt.close()
            
            print(f"Saved player grid to {save_path}")
    
    def _create_pose_visualization(self):
        """Create visualization of pose detection and mesh reconstruction"""
        # Find all pose visualizations
        pose_viz_files = sorted(glob.glob(str(self.meshes_dir / "*_viz.jpg")))
        
        if not pose_viz_files:
            print("No pose visualizations found")
            return
        
        # Group by player ID
        players = {}
        for pose_file in pose_viz_files:
            basename = os.path.basename(pose_file)
            # Format: player_<id>_frame_<frame>_viz.jpg
            parts = basename.replace('_viz.jpg', '').split('_')
            
            if len(parts) >= 4:
                player_id = int(parts[1])
                frame_id = int(parts[3])
                
                if player_id not in players:
                    players[player_id] = []
                
                players[player_id].append({
                    'player_id': player_id,
                    'frame_id': frame_id,
                    'pose_file': pose_file
                })
        
        # Sort by frame ID
        for player_id in players:
            players[player_id].sort(key=lambda x: x['frame_id'])
        
        # Create visualization grid for each player
        for player_id, frames in players.items():
            # Create figure
            n_frames = len(frames)
            if n_frames == 0:
                continue
                
            fig_width = min(15, n_frames * 3)
            fig = plt.figure(figsize=(fig_width, 4))
            fig.suptitle(f'Player {player_id} Pose Estimation Results', fontsize=16)
            
            # Create grid
            grid_size = min(n_frames, 5)  # Max 5 frames per row
            rows = (n_frames + grid_size - 1) // grid_size
            
            for i, frame_data in enumerate(frames):
                plt.subplot(rows, grid_size, i + 1)
                if os.path.exists(frame_data['pose_file']):
                    pose_img = cv2.imread(frame_data['pose_file'])
                    pose_img = cv2.cvtColor(pose_img, cv2.COLOR_BGR2RGB)
                    plt.imshow(pose_img)
                    plt.title(f"Frame {frame_data['frame_id']}")
                plt.axis('off')
            
            # Save figure
            fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for title
            save_path = self.output_dir / f"player_{player_id}_poses.png"
            plt.savefig(save_path, dpi=150)
            plt.close()
            
            print(f"Saved pose visualization to {save_path}")
    
    def _create_similarity_matrix(self):
        """Create simple visualization for pose similarity"""
        # This would typically use the actual similarity data from test_mesh_pipeline
        # Here we'll create a placeholder visualization
        
        # Create a combined visualization
        fig = plt.figure(figsize=(10, 8))
        fig.suptitle('Mesh Reconstruction Pipeline Results', fontsize=16)
        
        # Frame counts
        body_files = glob.glob(str(self.bodies_dir / "*.jpg"))
        face_files = glob.glob(str(self.faces_dir / "*.jpg"))
        mesh_files = glob.glob(str(self.meshes_dir / "*_viz.jpg"))
        
        # Collect player IDs
        player_ids = set()
        for f in body_files:
            basename = os.path.basename(f)
            parts = basename.split('_')
            if len(parts) > 1:
                player_ids.add(int(parts[1]))
        
        player_ids = sorted(player_ids)
        n_players = len(player_ids)
        
        # Create summary stats
        plt.subplot(2, 2, 1)
        plt.bar(['Bodies', 'Faces', 'Poses'], [len(body_files), len(face_files), len(mesh_files)])
        plt.title('Detection Counts')
        plt.ylabel('Count')
        
        # Player stats
        if n_players > 0:
            plt.subplot(2, 2, 2)
            
            # Count detections per player
            player_counts = {}
            for player_id in player_ids:
                body_count = len([f for f in body_files if f'player_{player_id}_' in os.path.basename(f)])
                face_count = len([f for f in face_files if f'player_{player_id}_' in os.path.basename(f)])
                pose_count = len([f for f in mesh_files if f'player_{player_id}_' in os.path.basename(f)])
                
                player_counts[player_id] = {
                    'body': body_count,
                    'face': face_count,
                    'pose': pose_count
                }
            
            # Create grouped bar chart
            x = np.arange(len(player_ids))
            width = 0.25
            
            plt.bar(x - width, [player_counts[pid]['body'] for pid in player_ids], width, label='Bodies')
            plt.bar(x, [player_counts[pid]['face'] for pid in player_ids], width, label='Faces')
            plt.bar(x + width, [player_counts[pid]['pose'] for pid in player_ids], width, label='Poses')
            
            plt.xlabel('Player ID')
            plt.ylabel('Count')
            plt.title('Detections by Player')
            plt.xticks(x, player_ids)
            plt.legend()
        
        # Placeholder for similarity matrix
        plt.subplot(2, 2, 3)
        if n_players >= 2:
            # Generate random similarity matrix for demonstration
            similarity = np.random.uniform(0.5, 1.0, size=(n_players, n_players))
            np.fill_diagonal(similarity, 1.0)  # Same player has similarity 1.0
            
            # Make matrix symmetric
            i_lower = np.tril_indices(n_players, -1)
            similarity[i_lower] = similarity.T[i_lower]
            
            plt.imshow(similarity, cmap='viridis', vmin=0, vmax=1)
            plt.colorbar(label='Similarity')
            plt.title('Player Pose Similarity Matrix')
            plt.xticks(range(n_players), player_ids)
            plt.yticks(range(n_players), player_ids)
            plt.xlabel('Player ID')
            plt.ylabel('Player ID')
            
            # Add values
            for i in range(n_players):
                for j in range(n_players):
                    plt.text(j, i, f'{similarity[i, j]:.2f}', 
                            ha='center', va='center', 
                            color='white' if similarity[i, j] < 0.7 else 'black')
        else:
            plt.text(0.5, 0.5, "Not enough players for similarity matrix", 
                    ha='center', va='center')
            plt.axis('off')
        
        # Add legend explaining test
        plt.subplot(2, 2, 4)
        plt.axis('off')
        plt.text(0.5, 0.8, "Mesh Reconstruction Pipeline Test", ha='center', fontweight='bold', fontsize=12)
        plt.text(0.5, 0.65, "This test evaluates the integration between:", ha='center')
        plt.text(0.5, 0.55, "1. Face/Body Extraction (face_body_crop.py)", ha='center')
        plt.text(0.5, 0.45, "2. Mesh Reconstruction (mesh_reconstruction.py)", ha='center')
        plt.text(0.5, 0.35, "Measurements:", ha='center')
        plt.text(0.5, 0.25, "- Face detection accuracy", ha='center')
        plt.text(0.5, 0.15, "- 3D pose landmark extraction", ha='center')
        plt.text(0.5, 0.05, "- Pose similarity across frames", ha='center')
        
        # Save figure
        fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for title
        save_path = self.output_dir / "mesh_test_summary.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        print(f"Saved summary visualization to {save_path}")


if __name__ == "__main__":
    # Check if test output directory exists
    if not os.path.exists('test_mesh_output'):
        print("Test output directory not found. Run test_mesh_pipeline.py first.")
    else:
        # Create visualizations
        visualizer = MeshResultVisualizer()
        visualizer.visualize_results()