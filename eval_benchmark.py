"""
Module 8: Evaluation and Benchmarking
Evaluate system performance using metrics and datasets
"""

import numpy as np
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import json
from pathlib import Path

class TrackingEvaluator:
    """Evaluate tracking and re-identification performance"""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset all metrics"""
        self.ground_truth = {}  # frame_id -> list of GT annotations
        self.predictions = {}   # frame_id -> list of predictions
        self.identity_switches = 0
        self.fragmentations = 0
        self.false_positives = 0
        self.false_negatives = 0
        
    def add_ground_truth(self, frame_id: int, annotations: List[Dict]):
        """
        Add ground truth annotations for a frame
        Args:
            frame_id: Frame number
            annotations: List of dicts with 'player_id', 'bbox'
        """
        self.ground_truth[frame_id] = annotations
        
    def add_predictions(self, frame_id: int, predictions: List[Dict]):
        """
        Add predictions for a frame
        Args:
            frame_id: Frame number
            predictions: List of dicts with 'player_id', 'bbox'
        """
        self.predictions[frame_id] = predictions
    
    @staticmethod
    def compute_iou(bbox1: List[float], bbox2: List[float]) -> float:
        """Compute IoU between two bboxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0
    
    def match_predictions_to_gt(self, gt_list: List[Dict], 
                                pred_list: List[Dict],
                                iou_threshold: float = 0.5) -> Dict:
        """
        Match predictions to ground truth using IoU
        Returns dict with matches, fps, fns
        """
        matches = []
        matched_gt = set()
        matched_pred = set()
        
        # For each prediction, find best GT match
        for pred_idx, pred in enumerate(pred_list):
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(gt_list):
                if gt_idx in matched_gt:
                    continue
                    
                iou = self.compute_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_gt_idx >= 0:
                matches.append({
                    'pred_idx': pred_idx,
                    'gt_idx': best_gt_idx,
                    'pred_id': pred['player_id'],
                    'gt_id': gt_list[best_gt_idx]['player_id'],
                    'iou': best_iou
                })
                matched_gt.add(best_gt_idx)
                matched_pred.add(pred_idx)
        
        fps = len(pred_list) - len(matched_pred)
        fns = len(gt_list) - len(matched_gt)
        
        return {
            'matches': matches,
            'false_positives': fps,
            'false_negatives': fns
        }
    
    def compute_identity_switches(self) -> int:
        """
        Compute number of identity switches
        An ID switch occurs when a GT track is assigned different predicted IDs
        """
        gt_to_pred_history = defaultdict(list)
        
        # Build history of GT->Pred ID mappings
        for frame_id in sorted(self.ground_truth.keys()):
            if frame_id not in self.predictions:
                continue
            
            gt_list = self.ground_truth[frame_id]
            pred_list = self.predictions[frame_id]
            
            result = self.match_predictions_to_gt(gt_list, pred_list)
            
            for match in result['matches']:
                gt_id = match['gt_id']
                pred_id = match['pred_id']
                gt_to_pred_history[gt_id].append((frame_id, pred_id))
        
        # Count switches
        switches = 0
        for gt_id, history in gt_to_pred_history.items():
            if len(history) < 2:
                continue
            
            for i in range(1, len(history)):
                prev_pred_id = history[i-1][1]
                curr_pred_id = history[i][1]
                
                if prev_pred_id != curr_pred_id:
                    switches += 1
        
        return switches
    
    def compute_mota(self) -> float:
        """
        Compute MOTA (Multiple Object Tracking Accuracy)
        MOTA = 1 - (FN + FP + IDSW) / GT
        """
        total_gt = 0
        total_fp = 0
        total_fn = 0
        
        for frame_id in sorted(self.ground_truth.keys()):
            if frame_id not in self.predictions:
                total_fn += len(self.ground_truth[frame_id])
                total_gt += len(self.ground_truth[frame_id])
                continue
            
            gt_list = self.ground_truth[frame_id]
            pred_list = self.predictions[frame_id]
            
            result = self.match_predictions_to_gt(gt_list, pred_list)
            
            total_fp += result['false_positives']
            total_fn += result['false_negatives']
            total_gt += len(gt_list)
        
        id_switches = self.compute_identity_switches()
        
        if total_gt == 0:
            return 0.0
        
        mota = 1 - (total_fn + total_fp + id_switches) / total_gt
        return max(0.0, mota)  # MOTA can be negative
    
    def compute_idf1(self) -> float:
        """
        Compute IDF1 (ID F1 Score)
        Measures identity preservation
        """
        idtp = 0  # ID True Positives
        idfn = 0  # ID False Negatives
        idfp = 0  # ID False Positives
        
        # Track GT ID -> Pred ID mappings
        gt_pred_pairs = defaultdict(lambda: defaultdict(int))
        
        for frame_id in sorted(self.ground_truth.keys()):
            if frame_id not in self.predictions:
                continue
            
            gt_list = self.ground_truth[frame_id]
            pred_list = self.predictions[frame_id]
            
            result = self.match_predictions_to_gt(gt_list, pred_list)
            
            for match in result['matches']:
                gt_id = match['gt_id']
                pred_id = match['pred_id']
                gt_pred_pairs[gt_id][pred_id] += 1
        
        # Compute IDTP
        for gt_id, pred_counts in gt_pred_pairs.items():
            # Find most common prediction ID for this GT ID
            most_common_pred = max(pred_counts.items(), key=lambda x: x[1])
            idtp += most_common_pred[1]
            
            # Other predictions are false positives
            for pred_id, count in pred_counts.items():
                if pred_id != most_common_pred[0]:
                    idfp += count
        
        # Count all GT and Pred occurrences
        total_gt_occurrences = sum(len(self.ground_truth[f]) 
                                   for f in self.ground_truth)
        total_pred_occurrences = sum(len(self.predictions[f]) 
                                     for f in self.predictions 
                                     if f in self.ground_truth)
        
        idfn = total_gt_occurrences - idtp
        idfp += total_pred_occurrences - idtp - idfp
        
        # IDF1 = 2*IDTP / (2*IDTP + IDFP + IDFN)
        denominator = 2 * idtp + idfp + idfn
        if denominator == 0:
            return 0.0
        
        return 2 * idtp / denominator
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute all tracking metrics"""
        mota = self.compute_mota()
        idf1 = self.compute_idf1()
        id_switches = self.compute_identity_switches()
        
        # Additional metrics
        total_gt = sum(len(self.ground_truth[f]) for f in self.ground_truth)
        total_pred = sum(len(self.predictions.get(f, [])) 
                        for f in self.ground_truth)
        
        return {
            'MOTA': mota,
            'IDF1': idf1,
            'ID_Switches': id_switches,
            'Total_GT': total_gt,
            'Total_Predictions': total_pred,
            'Precision': self._compute_precision(),
            'Recall': self._compute_recall()
        }
    
    def _compute_precision(self) -> float:
        """Compute detection precision"""
        total_tp = 0
        total_fp = 0
        
        for frame_id in self.ground_truth:
            if frame_id not in self.predictions:
                continue
            
            result = self.match_predictions_to_gt(
                self.ground_truth[frame_id],
                self.predictions[frame_id]
            )
            total_tp += len(result['matches'])
            total_fp += result['false_positives']
        
        if total_tp + total_fp == 0:
            return 0.0
        return total_tp / (total_tp + total_fp)
    
    def _compute_recall(self) -> float:
        """Compute detection recall"""
        total_tp = 0
        total_fn = 0
        
        for frame_id in self.ground_truth:
            if frame_id not in self.predictions:
                total_fn += len(self.ground_truth[frame_id])
                continue
            
            result = self.match_predictions_to_gt(
                self.ground_truth[frame_id],
                self.predictions[frame_id]
            )
            total_tp += len(result['matches'])
            total_fn += result['false_negatives']
        
        if total_tp + total_fn == 0:
            return 0.0
        return total_tp / (total_tp + total_fn)


class ReIDEvaluator:
    """Evaluate re-identification performance"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.correct_reids = 0
        self.incorrect_reids = 0
        self.reid_attempts = 0
        
    def evaluate_reid_event(self, predicted_id: int, true_id: int):
        """Evaluate a single re-ID event"""
        self.reid_attempts += 1
        if predicted_id == true_id:
            self.correct_reids += 1
        else:
            self.incorrect_reids += 1
    
    def compute_reid_accuracy(self) -> float:
        """Compute re-ID accuracy"""
        if self.reid_attempts == 0:
            return 0.0
        return self.correct_reids / self.reid_attempts
    
    def get_metrics(self) -> Dict[str, float]:
        """Get re-ID metrics"""
        return {
            'ReID_Accuracy': self.compute_reid_accuracy(),
            'Total_ReID_Attempts': self.reid_attempts,
            'Correct_ReIDs': self.correct_reids,
            'Incorrect_ReIDs': self.incorrect_reids
        }


class BenchmarkLoader:
    """Load benchmark datasets (SoccerNet, etc.)"""
    
    @staticmethod
    def load_soccernet_format(annotation_file: str) -> Dict:
        """
        Load SoccerNet-style annotations
        Expected format: JSON with frame-level annotations
        """
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        annotations = {}
        for frame_data in data.get('frames', []):
            frame_id = frame_data['frame_id']
            annotations[frame_id] = []
            
            for player in frame_data.get('players', []):
                annotations[frame_id].append({
                    'player_id': player['player_id'],
                    'bbox': player['bbox'],
                    'jersey': player.get('jersey_number', ''),
                    'team': player.get('team', '')
                })
        
        return annotations
    
    @staticmethod
    def create_sample_dataset(num_frames: int = 100,
                             num_players: int = 5) -> Dict:
        """Create synthetic test dataset"""
        annotations = {}
        
        for frame_id in range(num_frames):
            frame_anns = []
            
            for player_id in range(num_players):
                # Simulate player movement
                x = 100 + player_id * 150 + np.random.randint(-20, 20)
                y = 100 + frame_id * 2 + np.random.randint(-10, 10)
                
                frame_anns.append({
                    'player_id': player_id,
                    'bbox': [x, y, x + 80, y + 200],
                    'jersey': str(player_id + 1)
                })
            
            annotations[frame_id] = frame_anns
        
        return annotations


class PerformanceAnalyzer:
    """Analyze system performance and generate reports"""
    
    def __init__(self, output_dir: str = "evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def analyze_tracking_quality(self, evaluator: TrackingEvaluator) -> Dict:
        """Analyze tracking quality"""
        metrics = evaluator.compute_metrics()
        
        analysis = {
            'overall_quality': self._categorize_quality(metrics['MOTA']),
            'identity_preservation': self._categorize_quality(metrics['IDF1']),
            'metrics': metrics,
            'recommendations': []
        }
        
        # Generate recommendations
        if metrics['MOTA'] < 0.5:
            analysis['recommendations'].append(
                "Low MOTA: Consider improving detection quality or lowering IoU threshold"
            )
        
        if metrics['IDF1'] < 0.6:
            analysis['recommendations'].append(
                "Low IDF1: Improve re-identification features or adjust similarity threshold"
            )
        
        if metrics['ID_Switches'] > metrics['Total_GT'] * 0.1:
            analysis['recommendations'].append(
                "High ID switches: Check re-identification logic and tracking parameters"
            )
        
        return analysis
    
    @staticmethod
    def _categorize_quality(score: float) -> str:
        """Categorize metric score"""
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.6:
            return "Good"
        elif score >= 0.4:
            return "Fair"
        else:
            return "Poor"
    
    def generate_report(self, tracking_metrics: Dict,
                       reid_metrics: Dict,
                       output_name: str = "evaluation_report.json"):
        """Generate comprehensive evaluation report"""
        report = {
            'timestamp': str(np.datetime64('now')),
            'tracking_metrics': tracking_metrics,
            'reid_metrics': reid_metrics,
            'summary': {
                'overall_mota': tracking_metrics.get('MOTA', 0.0),
                'overall_idf1': tracking_metrics.get('IDF1', 0.0),
                'reid_accuracy': reid_metrics.get('ReID_Accuracy', 0.0)
            }
        }
        
        # Save report
        report_path = self.output_dir / output_name
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n{'='*60}")
        print("EVALUATION REPORT")
        print(f"{'='*60}")
        print(f"MOTA: {report['summary']['overall_mota']:.3f}")
        print(f"IDF1: {report['summary']['overall_idf1']:.3f}")
        print(f"Re-ID Accuracy: {report['summary']['reid_accuracy']:.3f}")
        print(f"{'='*60}\n")
        
        return report
    
    def plot_metrics_over_time(self, frame_metrics: List[Dict],
                               output_name: str = "metrics_plot.png"):
        """Plot metrics over time (requires matplotlib)"""
        try:
            import matplotlib.pyplot as plt
            
            frames = [m['frame_id'] for m in frame_metrics]
            motas = [m['mota'] for m in frame_metrics]
            
            plt.figure(figsize=(12, 6))
            plt.plot(frames, motas, label='MOTA')
            plt.xlabel('Frame')
            plt.ylabel('MOTA')
            plt.title('Tracking Performance Over Time')
            plt.legend()
            plt.grid(True)
            
            plot_path = self.output_dir / output_name
            plt.savefig(plot_path)
            print(f"Saved plot to {plot_path}")
            
        except ImportError:
            print("matplotlib not available for plotting")


# Example usage
if __name__ == "__main__":
    # Initialize evaluators
    tracking_eval = TrackingEvaluator()
    reid_eval = ReIDEvaluator()
    analyzer = PerformanceAnalyzer()
    
    # Load or create test data
    gt_annotations = BenchmarkLoader.create_sample_dataset(
        num_frames=50,
        num_players=5
    )
    
    # Simulate predictions (with some errors)
    for frame_id, gt_list in gt_annotations.items():
        # Add ground truth
        tracking_eval.add_ground_truth(frame_id, gt_list)
        
        # Create predictions (simulate 90% accuracy)
        predictions = []
        for gt in gt_list:
            if np.random.rand() > 0.1:  # 90% detection rate
                pred = gt.copy()
                # Occasionally switch IDs (simulate tracking errors)
                if np.random.rand() < 0.05:
                    pred['player_id'] = (gt['player_id'] + 1) % 5
                predictions.append(pred)
        
        tracking_eval.add_predictions(frame_id, predictions)
    
    # Compute metrics
    tracking_metrics = tracking_eval.compute_metrics()
    reid_metrics = reid_eval.get_metrics()
    
    # Analyze
    analysis = analyzer.analyze_tracking_quality(tracking_eval)
    
    # Generate report
    report = analyzer.generate_report(tracking_metrics, reid_metrics)
    
    print("\nRecommendations:")
    for rec in analysis.get('recommendations', []):
        print(f"  - {rec}")