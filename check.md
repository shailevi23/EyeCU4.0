
# PILOT — 10 epochs to confirm the dataset trains and to time a full run.
train('pilot', 'yolo26s.pt', imgsz=960, epochs=10, patience=0)

=== pilot: yolo26s.pt @ 960px, 10 epochs ===
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt to 'yolo26s.pt': 100% ━━━━━━━━━━━━ 19.5MB 174.5MB/s 0.1s
Ultralytics 8.4.116 🚀 Python-3.12.13 torch-2.11.0+cu128 CUDA:0 (Tesla T4, 14913MiB)
engine/trainer: agnostic_nms=False, amp=True, angle=1.0, augment=False, auto_augment=randaugment, batch=-1, bgr=0.0, box=7.5, cache=False, cfg=None, channels_last=False, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, cls_remap=True, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=/content/football_dataset/football.yaml, degrees=0.0, deterministic=True, device=, dfl=1.5, dgrad=0.5, dis=6.0, distill_model=None, dlam=1.0, dlog=1.0, dnn=False, dropout=0.0, dynamic=False, embed=None, end2end=None, epochs=10, erasing=0.4, exist_ok=True, fliplr=0.5, flipud=0.0, format=torchscript, fraction=1.0, freeze=None, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=960, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=yolo26s.pt, momentum=0.937, mosaic=1.0, multi_scale=0.0, name=pilot, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=0, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=/content/runs, quantize=None, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/content/runs/pilot, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.5, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=tracktrack.yaml, translate=0.1, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
Downloading https://ultralytics.com/assets/Arial.ttf to '/root/.config/Ultralytics/Arial.ttf': 100% ━━━━━━━━━━━━ 755.1KB 24.2MB/s 0.0s
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
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 100% ━━━━━━━━━━━━ 5.3MB 92.5MB/s 0.1s
AMP: checks passed ✅
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 1701.3±1111.6 MB/s, size: 153.4 KB)
train: Scanning /content/football_dataset/labels/train... 366 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 366/366 2.3Kit/s 0.2s
train: /content/football_dataset/images/train/clemson_vs__notre_dame_000177.jpg: 1 duplicate labels removed
train: /content/football_dataset/images/train/croatia_1-1_czechia_000140.jpg: 1 duplicate labels removed
train: /content/football_dataset/images/train/croatia_1-1_czechia_000200.jpg: 1 duplicate labels removed
train: /content/football_dataset/images/train/croatia_1-1_czechia_000675.jpg: 1 duplicate labels removed
train: /content/football_dataset/images/train/croatia_1-1_czechia_000883.jpg: 1 duplicate labels removed
train: /content/football_dataset/images/train/croatia_1-1_czechia_001200.jpg: 1 duplicate labels removed
train: /content/football_dataset/images/train/croatia_1-1_czechia_001360.jpg: 1 duplicate labels removed
train: /content/football_dataset/images/train/fc_barcelona_3_vs_1_atletico_de_mad_000170.jpg: 1 duplicate labels removed
train: /content/football_dataset/images/train/sunday_league_full_match_000300.jpg: 1 duplicate labels removed
train: /content/football_dataset/images/train/sunday_league_full_match_000725.jpg: 1 duplicate labels removed
train: /content/football_dataset/images/train/women_2_001000.jpg: 1 duplicate labels removed
train: /content/football_dataset/images/train/youth_5_000355.jpg: 1 duplicate labels removed
train: New cache created: /content/football_dataset/labels/train.cache
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, method='weighted_average', num_output_channels=3), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))
AutoBatch: Computing optimal batch size for imgsz=960 at 60.0% GPU memory utilization.
AutoBatch: CUDA:0 (Tesla T4) 14.56G total, 0.14G reserved, 0.12G allocated, 14.31G free
      Params      GFLOPs  GPU_mem (GB)  forward (ms) backward (ms)                   input                  output
     9950960       50.64         1.611         85.24           nan        (1, 3, 960, 960)                    list
     9950960       101.3         3.662         56.89           nan        (2, 3, 960, 960)                    list
     9950960       202.6         7.090          77.1           nan        (4, 3, 960, 960)                    list
     9950960       405.1        14.097           147           nan        (8, 3, 960, 960)                    list
CUDA out of memory. Tried to allocate 16.00 MiB. GPU 0 has a total capacity of 14.56 GiB of which 15.81 MiB is free. Including non-PyTorch memory, this process has 14.54 GiB memory in use. Of the allocated memory 13.90 GiB is allocated by PyTorch, and 504.07 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)
AutoBatch: Using batch-size 5 for CUDA:0 9.08G/14.56G (62%) ✅
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 1969.6±700.3 MB/s, size: 47.4 KB)
train: Scanning /content/football_dataset/labels/train.cache... 366 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 366/366 139.6Mit/s 0.0s
train: /content/football_dataset/images/train/clemson_vs__notre_dame_000177.jpg: 1 duplicate labels removed
train: /content/football_dataset/images/train/croatia_1-1_czechia_000140.jpg: 1 duplicate labels removed
train: /content/football_dataset/images/train/croatia_1-1_czechia_000200.jpg: 1 duplicate labels removed
train: /content/football_dataset/images/train/croatia_1-1_czechia_000675.jpg: 1 duplicate labels removed
train: /content/football_dataset/images/train/croatia_1-1_czechia_000883.jpg: 1 duplicate labels removed
train: /content/football_dataset/images/train/croatia_1-1_czechia_001200.jpg: 1 duplicate labels removed
train: /content/football_dataset/images/train/croatia_1-1_czechia_001360.jpg: 1 duplicate labels removed
train: /content/football_dataset/images/train/fc_barcelona_3_vs_1_atletico_de_mad_000170.jpg: 1 duplicate labels removed
train: /content/football_dataset/images/train/sunday_league_full_match_000300.jpg: 1 duplicate labels removed
train: /content/football_dataset/images/train/sunday_league_full_match_000725.jpg: 1 duplicate labels removed
train: /content/football_dataset/images/train/women_2_001000.jpg: 1 duplicate labels removed
train: /content/football_dataset/images/train/youth_5_000355.jpg: 1 duplicate labels removed
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, method='weighted_average', num_output_channels=3), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 408.7±66.9 MB/s, size: 55.5 KB)
val: Scanning /content/football_dataset/labels/val... 85 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 85/85 1.5Kit/s 0.1s
val: /content/football_dataset/images/val/youth_7_000320.jpg: 1 duplicate labels removed
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
       1/10      3.76G      1.207       6.68   0.003814          3        960: 100% ━━━━━━━━━━━━ 74/74 2.1it/s 34.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 1.1it/s 8.5s
                   all         85       1270      0.243      0.276      0.266      0.184

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
       2/10      3.94G      1.156      2.321   0.002688          8        960: 100% ━━━━━━━━━━━━ 74/74 3.8it/s 19.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 8.6it/s 1.0s
                   all         85       1270      0.755      0.457      0.441      0.294

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
       3/10      3.94G      1.226      1.378   0.002784         11        960: 100% ━━━━━━━━━━━━ 74/74 3.8it/s 19.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 6.7it/s 1.3s
                   all         85       1270      0.829      0.474      0.511      0.339

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
       4/10      3.94G       1.16       1.06   0.002737         13        960: 100% ━━━━━━━━━━━━ 74/74 3.7it/s 20.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 8.5it/s 1.1s
                   all         85       1270      0.804      0.449      0.512      0.352

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
       5/10      3.94G      1.153     0.8736   0.002721         20        960: 100% ━━━━━━━━━━━━ 74/74 3.7it/s 19.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 6.7it/s 1.3s
                   all         85       1270       0.83      0.476      0.544      0.372

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
       6/10      3.94G      1.177     0.7794   0.002883         18        960: 100% ━━━━━━━━━━━━ 74/74 3.6it/s 20.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 8.4it/s 1.1s
                   all         85       1270      0.868      0.473      0.561       0.39

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
       7/10      3.94G      1.129     0.7072    0.00274         10        960: 100% ━━━━━━━━━━━━ 74/74 3.7it/s 20.1s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 6.8it/s 1.3s
                   all         85       1270      0.895      0.489      0.566      0.388

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
       8/10      3.94G      1.142     0.6609   0.002591         15        960: 100% ━━━━━━━━━━━━ 74/74 3.6it/s 20.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 8.3it/s 1.1s
                   all         85       1270      0.605      0.585      0.592      0.415

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
       9/10      3.94G      1.049     0.6217   0.002374          3        960: 100% ━━━━━━━━━━━━ 74/74 3.7it/s 19.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 7.5it/s 1.2s
                   all         85       1270      0.734      0.531      0.607      0.431

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
      10/10      3.94G      1.042     0.5987   0.002359         16        960: 100% ━━━━━━━━━━━━ 74/74 3.6it/s 20.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 8.5it/s 1.1s
                   all         85       1270      0.727      0.526      0.621       0.45

10 epochs completed in 0.067 hours.
Optimizer stripped from /content/runs/pilot/weights/last.pt, 20.3MB
Optimizer stripped from /content/runs/pilot/weights/best.pt, 20.3MB

Validating /content/runs/pilot/weights/best.pt...
Ultralytics 8.4.116 🚀 Python-3.12.13 torch-2.11.0+cu128 CUDA:0 (Tesla T4, 14913MiB)
YOLO26s summary (fused): 122 layers, 9,466,728 parameters, 0 gradients, 20.8 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 3.4it/s 2.7s
                   all         85       1270      0.745      0.519      0.621       0.45
                player         85       1068      0.913      0.906      0.945       0.71
            goalkeeper         33         33      0.393      0.121      0.251      0.194
               referee         67        103       0.75      0.496       0.65      0.479
                  ball         65         66      0.924      0.554      0.639      0.416
Speed: 0.5ms preprocess, 9.7ms inference, 0.0ms loss, 0.5ms postprocess per image
Results saved to /content/runs/pilot
Ultralytics 8.4.116 🚀 Python-3.12.13 torch-2.11.0+cu128 CUDA:0 (Tesla T4, 14913MiB)
YOLO26s summary (fused): 122 layers, 9,466,728 parameters, 0 gradients, 20.8 GFLOPs
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 1280.6±287.8 MB/s, size: 57.3 KB)
val: Scanning /content/football_dataset/labels/val.cache... 85 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 85/85 22.3Mit/s 0.0s
val: /content/football_dataset/images/val/youth_7_000320.jpg: 1 duplicate labels removed
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 1.4it/s 4.3s
                   all         85       1270      0.728      0.526      0.622       0.45
Speed: 5.3ms preprocess, 14.1ms inference, 0.0ms loss, 1.0ms postprocess per image
Results saved to /content/runs/detect/val
{
  "weights": "yolo26s.pt",
  "imgsz": 960,
  "epochs": 10,
  "train_minutes": 4.8,
  "best_pt": "/content/runs/pilot/weights/best.pt",
  "split": "val",
  "mAP50": 0.6218934883408396,
  "mAP50_95": 0.45046802244540735,
  "precision_mean": 0.7277031547841182,
  "recall_mean": 0.526230519063907,
  "per_class": {
    "player": {
      "precision": 0.9043862961498765,
      "recall": 0.9091760299625468,
      "mAP50": 0.9439019406136341,
      "mAP50_95": 0.7100612055346406
    },
    "goalkeeper": {
      "precision": 0.3673725064227414,
      "recall": 0.12121212121212122,
      "mAP50": 0.2548317290375591,
      "mAP50_95": 0.19784830672897935
    },
    "referee": {
      "precision": 0.7129583579556314,
      "recall": 0.5048543689320388,
      "mAP50": 0.6507366874238276,
      "mAP50_95": 0.48061821205879784
    },
    "ball": {
      "precision": 0.9260954586082235,
      "recall": 0.5696795561489215,
      "mAP50": 0.6381035962883372,
      "mAP50_95": 0.4133443654592119
    }
  },
  "inference_ms": 14.14,
  "total_ms_per_image": 20.45,
  "fps": 48.9
}
{'weights': 'yolo26s.pt',
 'imgsz': 960,
 'epochs': 10,
 'train_minutes': 4.8,
 'best_pt': '/content/runs/pilot/weights/best.pt',
 'split': 'val',
 'mAP50': 0.6218934883408396,
 'mAP50_95': 0.45046802244540735,
 'precision_mean': 0.7277031547841182,
 'recall_mean': 0.526230519063907,
 'per_class': {'player': {'precision': 0.9043862961498765,
   'recall': 0.9091760299625468,
   'mAP50': 0.9439019406136341,
   'mAP50_95': 0.7100612055346406},
  'goalkeeper': {'precision': 0.3673725064227414,
   'recall': 0.12121212121212122,
   'mAP50': 0.2548317290375591,
   'mAP50_95': 0.19784830672897935},
  'referee': {'precision': 0.7129583579556314,
   'recall': 0.5048543689320388,
   'mAP50': 0.6507366874238276,
   'mAP50_95': 0.48061821205879784},
  'ball': {'precision': 0.9260954586082235,
   'recall': 0.5696795561489215,
   'mAP50': 0.6381035962883372,
   'mAP50_95': 0.4133443654592119}},
 'inference_ms': 14.14,
 'total_ms_per_image': 20.45,
 'fps': 48.9}