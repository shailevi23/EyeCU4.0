=== pilot: yolo26s.pt @ 960px, 10 epochs ===
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt to 'yolo26s.pt': 100% ━━━━━━━━━━━━ 19.5MB 269.8MB/s 0.1s
Ultralytics 8.4.116 🚀 Python-3.12.13 torch-2.11.0+cu128 CUDA:0 (Tesla T4, 14913MiB)
engine/trainer: agnostic_nms=False, amp=True, angle=1.0, augment=False, auto_augment=randaugment, batch=-1, bgr=0.0, box=7.5, cache=False, cfg=None, channels_last=False, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, cls_remap=True, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=/content/football_dataset/football.yaml, degrees=0.0, deterministic=True, device=, dfl=1.5, dgrad=0.5, dis=6.0, distill_model=None, dlam=1.0, dlog=1.0, dnn=False, dropout=0.0, dynamic=False, embed=None, end2end=None, epochs=10, erasing=0.4, exist_ok=True, fliplr=0.5, flipud=0.0, format=torchscript, fraction=1.0, freeze=None, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=960, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=yolo26s.pt, momentum=0.937, mosaic=1.0, multi_scale=0.0, name=pilot, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=0, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=/content/runs, quantize=None, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/content/runs/pilot, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.5, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=tracktrack.yaml, translate=0.1, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
Downloading https://ultralytics.com/assets/Arial.ttf to '/root/.config/Ultralytics/Arial.ttf': 100% ━━━━━━━━━━━━ 755.1KB 132.9MB/s 0.0s
Overriding model.yaml nc=80 with nc=4

                   from  n    params  module                                       arguments                     
  0                  -1  1       928  ultralytics.nn.modules.conv.Conv             [3, 32, 3, 2]                 
  1                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]                
  2                  -1  1     26080  ultralytics.nn.modules.block.C3k2            [64, 128, 1, False, 0.25]     
  3                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
  4                  -1  1    103360  ultralytics.nn.modules.block.C3k2            [128, 256, 1, False, 0.25]    
  5                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
  6                  -1  1    346112  ultralytics.nn.modules.block.C3k2            [256, 256, 1, True]           
  7                  -1  1   1180672  ultralytics.nn.modules.conv.Conv             [256, 512, 3, 2]              
  8                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           
  9                  -1  1    656896  ultralytics.nn.modules.block.SPPF            [512, 512, 5, 3, True]        
 10                  -1  1    990976  ultralytics.nn.modules.block.C2PSA           [512, 512, 1]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  1    477184  ultralytics.nn.modules.block.C3k2            [768, 256, 1, True]           
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  1    136192  ultralytics.nn.modules.block.C3k2            [512, 128, 1, True]           
 17                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  1    378880  ultralytics.nn.modules.block.C3k2            [384, 256, 1, True]           
 20                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  1   1843712  ultralytics.nn.modules.block.C3k2            [768, 512, 1, True, 0.5, True]
 23        [16, 19, 22]  1    934960  ultralytics.nn.modules.head.Detect           [4, 1, True, [128, 256, 512]] 
YOLO26s summary: 260 layers, 9,950,960 parameters, 9,950,960 gradients, 22.8 GFLOPs

Transferred 696/708 items from pretrained weights
AMP: running Automatic Mixed Precision (AMP) checks...
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 100% ━━━━━━━━━━━━ 5.3MB 232.5MB/s 0.0s
AMP: checks passed ✅
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2344.6±1380.1 MB/s, size: 105.7 KB)
train: Scanning /content/football_dataset/labels/train... 823 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 823/823 1.9Kit/s 0.4s
train: New cache created: /content/football_dataset/labels/train.cache
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, method='weighted_average', num_output_channels=3), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))
AutoBatch: Computing optimal batch size for imgsz=960 at 60.0% GPU memory utilization.
AutoBatch: CUDA:0 (Tesla T4) 14.56G total, 0.14G reserved, 0.12G allocated, 14.31G free
      Params      GFLOPs  GPU_mem (GB)  forward (ms) backward (ms)                   input                  output
     9950960       50.64         1.615         75.95           nan        (1, 3, 960, 960)                    list
     9950960       101.3         3.668         48.52           nan        (2, 3, 960, 960)                    list
     9950960       202.6         7.105         77.05           nan        (4, 3, 960, 960)                    list
     9950960       405.1        14.124         142.2           nan        (8, 3, 960, 960)                    list
CUDA out of memory. Tried to allocate 16.00 MiB. GPU 0 has a total capacity of 14.56 GiB of which 15.81 MiB is free. Including non-PyTorch memory, this process has 14.54 GiB memory in use. Of the allocated memory 13.90 GiB is allocated by PyTorch, and 504.07 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)
AutoBatch: Using batch-size 5 for CUDA:0 9.10G/14.56G (62%) ✅
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2759.9±1757.5 MB/s, size: 147.1 KB)
train: Scanning /content/football_dataset/labels/train.cache... 823 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 823/823 191.8Mit/s 0.0s
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, method='weighted_average', num_output_channels=3), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 337.0±227.5 MB/s, size: 58.6 KB)
val: Scanning /content/football_dataset/labels/val... 208 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 208/208 1.4Kit/s 0.1s
val: New cache created: /content/football_dataset/labels/val.cache
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.00125, momentum=0.9) with parameter groups 114 weight(decay=0.0), 126 weight(decay=0.0005078125), 126 bias(decay=0.0)
Plotting labels to /content/runs/pilot/labels.jpg... 
Image sizes 960 train, 960 val
Using 2 dataloader workers
Logging results to /content/runs/pilot
Starting training for 10 epochs...
Closing dataloader mosaic
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, method='weighted_average', num_output_channels=3), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
       1/10       3.7G      1.305      3.951   0.001888         63        960: 100% ━━━━━━━━━━━━ 165/165 2.4it/s 1:08
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 21/21 2.3it/s 9.1s
                   all        208       2973       0.45       0.41      0.447      0.279

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
       2/10      3.88G      1.274      1.134   0.001969         59        960: 100% ━━━━━━━━━━━━ 165/165 3.6it/s 46.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 21/21 6.1it/s 3.4s
                   all        208       2973      0.612      0.569      0.546      0.325

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
       3/10      3.88G      1.239      0.768    0.00193         48        960: 100% ━━━━━━━━━━━━ 165/165 3.5it/s 46.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 21/21 7.7it/s 2.7s
                   all        208       2973      0.653      0.484      0.544      0.341

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
       4/10      3.88G      1.235      0.674   0.001824         46        960: 100% ━━━━━━━━━━━━ 165/165 3.5it/s 46.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 21/21 7.9it/s 2.7s
                   all        208       2973      0.649      0.598        0.6      0.378

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
       5/10      3.88G      1.217     0.6105   0.001924         47        960: 100% ━━━━━━━━━━━━ 165/165 3.5it/s 47.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 21/21 7.0it/s 3.0s
                   all        208       2973      0.719      0.554      0.586      0.367

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
       6/10      3.88G      1.165     0.5465   0.001815         58        960: 100% ━━━━━━━━━━━━ 165/165 3.5it/s 46.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 21/21 7.9it/s 2.7s
                   all        208       2973      0.777      0.578      0.669      0.425

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
       7/10      3.88G      1.129     0.4989   0.001728         49        960: 100% ━━━━━━━━━━━━ 165/165 3.5it/s 47.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 21/21 7.1it/s 2.9s
                   all        208       2973      0.759      0.683      0.722      0.459

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
       8/10      3.88G       1.13     0.4785    0.00165         65        960: 100% ━━━━━━━━━━━━ 165/165 3.5it/s 47.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 21/21 7.3it/s 2.9s
                   all        208       2973      0.728      0.657      0.683      0.439

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
       9/10      3.88G      1.091      0.452   0.001569         69        960: 100% ━━━━━━━━━━━━ 165/165 3.5it/s 46.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 21/21 8.0it/s 2.6s
                   all        208       2973      0.736      0.639      0.683      0.439

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
      10/10      3.88G      1.083     0.4515   0.001552         45        960: 100% ━━━━━━━━━━━━ 165/165 3.5it/s 46.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 21/21 6.4it/s 3.3s
                   all        208       2973      0.744      0.644      0.692      0.444

10 epochs completed in 0.148 hours.
Optimizer stripped from /content/runs/pilot/weights/last.pt, 20.3MB
Optimizer stripped from /content/runs/pilot/weights/best.pt, 20.3MB

Validating /content/runs/pilot/weights/best.pt...
Ultralytics 8.4.116 🚀 Python-3.12.13 torch-2.11.0+cu128 CUDA:0 (Tesla T4, 14913MiB)
YOLO26s summary (fused): 122 layers, 9,466,728 parameters, 0 gradients, 20.8 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 21/21 5.3it/s 4.0s
                   all        208       2973       0.76      0.683      0.722      0.461
                player        204       2490      0.876      0.926      0.953      0.667
            goalkeeper        114        115      0.773      0.574      0.653      0.444
               referee        141        257      0.611      0.746      0.733      0.466
                  ball        111        111      0.779      0.486      0.548      0.267
Speed: 0.6ms preprocess, 7.1ms inference, 0.0ms loss, 0.5ms postprocess per image
Results saved to /content/runs/pilot
Ultralytics 8.4.116 🚀 Python-3.12.13 torch-2.11.0+cu128 CUDA:0 (Tesla T4, 14913MiB)
YOLO26s summary (fused): 122 layers, 9,466,728 parameters, 0 gradients, 20.8 GFLOPs
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 1902.3±619.8 MB/s, size: 55.1 KB)
val: Scanning /content/football_dataset/labels/val.cache... 208 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 208/208 87.2Mit/s 0.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 13/13 1.8it/s 7.2s
                   all        208       2973      0.757      0.683      0.722      0.459
Speed: 6.0ms preprocess, 13.4ms inference, 0.0ms loss, 0.7ms postprocess per image
Results saved to /content/runs/detect/val
{
  "weights": "yolo26s.pt",
  "imgsz": 960,
  "epochs": 10,
  "train_minutes": 9.7,
  "best_pt": "/content/runs/pilot/weights/best.pt",
  "split": "val",
  "mAP50": 0.7216602947411956,
  "mAP50_95": 0.4589215267093346,
  "precision_mean": 0.7573041788500373,
  "recall_mean": 0.6833964149244217,
  "per_class": {
    "player": {
      "precision": 0.8758553715776065,
      "recall": 0.9261044176706827,
      "mAP50": 0.9531844121902207,
      "mAP50_95": 0.6676429310504078
    },
    "goalkeeper": {
      "precision": 0.7720674802581837,
      "recall": 0.5739130434782609,
      "mAP50": 0.6535301107072773,
      "mAP50_95": 0.43450635614263744
    },
    "referee": {
      "precision": 0.60545090858789,
      "recall": 0.7470817120622568,
      "mAP50": 0.7329646809603116,
      "mAP50_95": 0.46632064153356756
    },
    "ball": {
      "precision": 0.7758429549764688,
      "recall": 0.4864864864864865,
      "mAP50": 0.5469619751069729,
      "mAP50_95": 0.2672161781107258
    }
  },
  "inference_ms": 13.43,
  "total_ms_per_image": 20.11,
  "fps": 49.7
}
{'weights': 'yolo26s.pt',
 'imgsz': 960,
 'epochs': 10,
 'train_minutes': 9.7,
 'best_pt': '/content/runs/pilot/weights/best.pt',
 'split': 'val',
 'mAP50': 0.7216602947411956,
 'mAP50_95': 0.4589215267093346,
 'precision_mean': 0.7573041788500373,
 'recall_mean': 0.6833964149244217,
 'per_class': {'player': {'precision': 0.8758553715776065,
   'recall': 0.9261044176706827,
   'mAP50': 0.9531844121902207,
   'mAP50_95': 0.6676429310504078},
  'goalkeeper': {'precision': 0.7720674802581837,
   'recall': 0.5739130434782609,
   'mAP50': 0.6535301107072773,
   'mAP50_95': 0.43450635614263744},
  'referee': {'precision': 0.60545090858789,
   'recall': 0.7470817120622568,
   'mAP50': 0.7329646809603116,
   'mAP50_95': 0.46632064153356756},
  'ball': {'precision': 0.7758429549764688,
   'recall': 0.4864864864864865,
   'mAP50': 0.5469619751069729,
   'mAP50_95': 0.2672161781107258}},
 'inference_ms': 13.43,
 'total_ms_per_image': 20.11,
 'fps': 49.7}




 LLM ADVISOR ANSEWR:

 This looks **good enough to continue to the full Experiment A**.

The important thing is that you're now evaluating on the **real 208-frame validation set** rather than the old temporary 85-frame val. The pilot trained cleanly on 823 train images and validated on all 208 images. 

After only 10 epochs:

* **Player recall:** 92.6% — excellent
* **Goalkeeper recall:** 57.4% — reasonable
* **Referee recall:** 74.7% — good
* **Ball recall:** **48.6% — clearly the weak point**
* overall mAP50-95: **0.459**
* ~**49.7 FPS** in this validation timing. 

One important point: don't compare the 48.6% ball recall directly with the old ~59% result and conclude the model got worse. **The validation set changed completely**, so this is now a harder and much more meaningful measurement.

### My decision: GO ✅

Run the proper A training now:

```python
train(
    'A_yolo26s_960_realval',
    'yolo26s.pt',
    imgsz=960,
    epochs=80
)
```

Keep `patience=20` as your function default. The AutoBatch OOM probe is harmless; it recovered and selected batch 5 normally. 

When A finishes, **then** we decide whether 1280 is worth testing. Right now the main question is whether 80 epochs can lift that **ball recall** while keeping referee/goalkeeper generalization strong.
