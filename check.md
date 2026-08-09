=== A_yolo26s_960_realval: yolo26s.pt @ 960px, 80 epochs ===
Ultralytics 8.4.116 🚀 Python-3.12.13 torch-2.11.0+cu128 CUDA:0 (Tesla T4, 14913MiB)
engine/trainer: agnostic_nms=False, amp=True, angle=1.0, augment=False, auto_augment=randaugment, batch=-1, bgr=0.0, box=7.5, cache=False, cfg=None, channels_last=False, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, cls_remap=True, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=/content/football_dataset/football.yaml, degrees=0.0, deterministic=True, device=, dfl=1.5, dgrad=0.5, dis=6.0, distill_model=None, dlam=1.0, dlog=1.0, dnn=False, dropout=0.0, dynamic=False, embed=None, end2end=None, epochs=80, erasing=0.4, exist_ok=True, fliplr=0.5, flipud=0.0, format=torchscript, fraction=1.0, freeze=None, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=960, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=yolo26s.pt, momentum=0.937, mosaic=1.0, multi_scale=0.0, name=A_yolo26s_960_realval, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=20, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=/content/runs, quantize=None, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/content/runs/A_yolo26s_960_realval, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.5, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=tracktrack.yaml, translate=0.1, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
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
AMP: checks passed ✅
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2134.1±1116.0 MB/s, size: 109.1 KB)
train: Scanning /content/football_dataset/labels/train.cache... 823 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 823/823 181.7Mit/s 0.0s
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, method='weighted_average', num_output_channels=3), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))
AutoBatch: Computing optimal batch size for imgsz=960 at 60.0% GPU memory utilization.
AutoBatch: CUDA:0 (Tesla T4) 14.56G total, 2.01G reserved, 0.38G allocated, 12.18G free
      Params      GFLOPs  GPU_mem (GB)  forward (ms) backward (ms)                   input                  output
     9950960       50.64         1.697          63.2           nan        (1, 3, 960, 960)                    list
     9950960       101.3         3.766         64.24           nan        (2, 3, 960, 960)                    list
     9950960       202.6         7.195         80.56           nan        (4, 3, 960, 960)                    list
     9950960       405.1        14.227         112.1           nan        (8, 3, 960, 960)                    list
CUDA out of memory. Tried to allocate 30.00 MiB. GPU 0 has a total capacity of 14.56 GiB of which 1.81 MiB is free. Including non-PyTorch memory, this process has 14.56 GiB memory in use. Of the allocated memory 13.86 GiB is allocated by PyTorch, and 511.64 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)
AutoBatch: Using batch-size 3 for CUDA:0 7.78G/14.56G (53%) ✅
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2573.0±1096.7 MB/s, size: 139.4 KB)
train: Scanning /content/football_dataset/labels/train.cache... 823 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 823/823 313.8Mit/s 0.0s
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, method='weighted_average', num_output_channels=3), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 656.8±419.4 MB/s, size: 53.8 KB)
val: Scanning /content/football_dataset/labels/val.cache... 208 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 208/208 9.9Mit/s 0.0s
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.00125, momentum=0.9) with parameter groups 114 weight(decay=0.0), 126 weight(decay=0.0004921875), 126 bias(decay=0.0)
Plotting labels to /content/runs/A_yolo26s_960_realval/labels.jpg... 
Image sizes 960 train, 960 val
Using 2 dataloader workers
Logging results to /content/runs/A_yolo26s_960_realval
Starting training for 80 epochs...

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
       1/80      2.46G      1.292      2.523    0.00174         32        960: 100% ━━━━━━━━━━━━ 275/275 4.1it/s 1:07
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 35/35 3.4it/s 10.4s
                   all        208       2973      0.704      0.429      0.455      0.286

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
       2/80      2.46G       1.28     0.9852   0.001789         70        960: 100% ━━━━━━━━━━━━ 275/275 4.6it/s 1:00
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 35/35 9.4it/s 3.7s
                   all        208       2973      0.646      0.579      0.585      0.358

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
       3/80      2.46G      1.274     0.7826   0.001807         48        960: 100% ━━━━━━━━━━━━ 275/275 4.6it/s 59.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 35/35 11.1it/s 3.2s
                   all        208       2973      0.743      0.587      0.658      0.413

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
       4/80      2.46G      1.263     0.6886   0.001722          9        960: 100% ━━━━━━━━━━━━ 275/275 4.6it/s 59.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 35/35 12.1it/s 2.9s
                   all        208       2973      0.747      0.621      0.659      0.412

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
       5/80      2.46G      1.218     0.6438   0.001733         47        960: 100% ━━━━━━━━━━━━ 275/275 4.6it/s 1:00
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 35/35 10.2it/s 3.4s
                   all        208       2973      0.674      0.598      0.632      0.401

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
       6/80      2.46G       1.21     0.6049   0.001684         41        960: 100% ━━━━━━━━━━━━ 275/275 4.5it/s 1:01
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 35/35 11.9it/s 3.0s
                   all        208       2973      0.699      0.636      0.665      0.424

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
       7/80      2.46G       1.19     0.5673   0.001625         44        960: 100% ━━━━━━━━━━━━ 275/275 4.5it/s 1:00
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 35/35 12.1it/s 2.9s
                   all        208       2973      0.739      0.617      0.656      0.404

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
       8/80      2.46G      1.203     0.5642   0.001732         19        960: 100% ━━━━━━━━━━━━ 275/275 4.5it/s 1:01
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 35/35 8.9it/s 3.9s
                   all        208       2973      0.713       0.65      0.715      0.447

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
       9/80      2.46G       1.18     0.5455    0.00163         19        960: 100% ━━━━━━━━━━━━ 275/275 4.5it/s 1:01
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 35/35 11.1it/s 3.2s
                   all        208       2973       0.77      0.623      0.682      0.439

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
      10/80      2.46G      1.171     0.5481   0.001727         30        960: 100% ━━━━━━━━━━━━ 275/275 4.5it/s 1:01
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 35/35 11.9it/s 2.9s
                   all        208       2973      0.728      0.671      0.686      0.434

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
      11/80      2.46G      1.161     0.5159   0.001662         51        960: 100% ━━━━━━━━━━━━ 275/275 4.5it/s 1:01
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 35/35 8.6it/s 4.0s
                   all        208       2973      0.751      0.615      0.711      0.446

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
      12/80      2.46G      1.155     0.5093   0.001653         48        960: 100% ━━━━━━━━━━━━ 275/275 4.5it/s 1:02
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 35/35 11.9it/s 2.9s
                   all        208       2973      0.792      0.678      0.739      0.474

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
      13/80      2.46G      1.163     0.5018   0.001598         39        960: 100% ━━━━━━━━━━━━ 275/275 4.6it/s 59.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 35/35 12.2it/s 2.9s
                   all        208       2973      0.782      0.632      0.693      0.455

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
      14/80      2.46G      1.154     0.4979   0.001603         67        960: 100% ━━━━━━━━━━━━ 275/275 4.6it/s 59.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 35/35 9.9it/s 3.5s
                   all        208       2973      0.689      0.655      0.669      0.431

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
      15/80      2.46G      1.144     0.4971   0.001626         11        960: 100% ━━━━━━━━━━━━ 275/275 4.6it/s 59.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 35/35 10.8it/s 3.3s
                   all        208       2973      0.788      0.654      0.708      0.453

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
      16/80      2.46G      1.154     0.4942   0.001623         14        960: 100% ━━━━━━━━━━━━ 275/275 4.6it/s 59.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 35/35 12.1it/s 2.9s
                   all        208       2973      0.746      0.656      0.704      0.447

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
      17/80      2.46G      1.135     0.4879   0.001529         60        960: 100% ━━━━━━━━━━━━ 275/275 4.6it/s 1:00
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 35/35 10.8it/s 3.2s
                   all        208       2973       0.71      0.624       0.67      0.421

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
      18/80      2.46G      1.141     0.4906   0.001601         20        960: 100% ━━━━━━━━━━━━ 275/275 4.6it/s 59.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 35/35 9.5it/s 3.7s
                   all        208       2973      0.761       0.63      0.684      0.417

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
      19/80      2.46G      1.124     0.4775   0.001607         58        960: 100% ━━━━━━━━━━━━ 275/275 4.7it/s 58.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 35/35 12.0it/s 2.9s
                   all        208       2973      0.785      0.642      0.702      0.444

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
      20/80      2.46G      1.133     0.4697   0.001541         40        960: 100% ━━━━━━━━━━━━ 275/275 4.6it/s 59.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 35/35 12.3it/s 2.8s
                   all        208       2973      0.789      0.656      0.727      0.464

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
      21/80      2.46G      1.104     0.4518   0.001544         23        960: 100% ━━━━━━━━━━━━ 275/275 4.7it/s 58.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 35/35 9.5it/s 3.7s
                   all        208       2973      0.727      0.642      0.689      0.437

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
      22/80      2.46G      1.106     0.4481   0.001565         28        960: 100% ━━━━━━━━━━━━ 275/275 4.6it/s 60.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 35/35 12.1it/s 2.9s
                   all        208       2973      0.763      0.668      0.706      0.439

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
      23/80      2.46G        1.1      0.437   0.001527         16        960: 100% ━━━━━━━━━━━━ 275/275 4.6it/s 1:00
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 35/35 12.1it/s 2.9s
                   all        208       2973      0.742      0.652      0.672      0.427

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
      24/80      2.46G      1.086     0.4245   0.001451         11        960: 100% ━━━━━━━━━━━━ 275/275 4.6it/s 59.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 35/35 9.0it/s 3.9s
                   all        208       2973      0.796      0.663      0.698      0.446

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
      25/80      2.46G       1.09     0.4376   0.001545         26        960: 100% ━━━━━━━━━━━━ 275/275 4.5it/s 1:01
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 35/35 12.2it/s 2.9s
                   all        208       2973      0.766      0.658      0.694      0.444

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
      26/80      2.46G      1.089     0.4246   0.001478         29        960: 100% ━━━━━━━━━━━━ 275/275 4.6it/s 59.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 35/35 12.1it/s 2.9s
                   all        208       2973       0.79      0.645      0.692      0.444

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
      27/80      2.46G      1.089     0.4273   0.001504         19        960: 100% ━━━━━━━━━━━━ 275/275 4.6it/s 59.9s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 35/35 9.1it/s 3.8s
                   all        208       2973      0.766      0.683      0.708      0.447

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
      28/80      2.46G       1.07     0.4181   0.001484         21        960: 100% ━━━━━━━━━━━━ 275/275 4.6it/s 1:00
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 35/35 12.1it/s 2.9s
                   all        208       2973      0.792      0.686      0.718       0.46

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
      29/80      2.46G      1.078     0.4226   0.001477         49        960: 100% ━━━━━━━━━━━━ 275/275 4.6it/s 59.7s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 35/35 12.2it/s 2.9s
                   all        208       2973      0.809      0.706      0.711      0.461

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
      30/80      2.46G      1.076     0.4075   0.001401         17        960: 100% ━━━━━━━━━━━━ 275/275 4.6it/s 59.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 35/35 10.2it/s 3.4s
                   all        208       2973      0.833      0.667      0.734      0.471

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
      31/80      2.46G      1.083       0.41   0.001399         21        960: 100% ━━━━━━━━━━━━ 275/275 4.6it/s 59.5s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 35/35 9.9it/s 3.5s
                   all        208       2973      0.794      0.686      0.729      0.472

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
      32/80      2.46G      1.058     0.4074   0.001418         33        960: 100% ━━━━━━━━━━━━ 275/275 4.6it/s 59.6s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 35/35 12.2it/s 2.9s
                   all        208       2973      0.817      0.645      0.735      0.465
EarlyStopping: Training stopped early as no improvement observed in last 20 epochs. Best results observed at epoch 12, best model saved as best.pt.
To update EarlyStopping(patience=20) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.

32 epochs completed in 0.573 hours.
Optimizer stripped from /content/runs/A_yolo26s_960_realval/weights/last.pt, 20.3MB
Optimizer stripped from /content/runs/A_yolo26s_960_realval/weights/best.pt, 20.3MB

Validating /content/runs/A_yolo26s_960_realval/weights/best.pt...
Ultralytics 8.4.116 🚀 Python-3.12.13 torch-2.11.0+cu128 CUDA:0 (Tesla T4, 14913MiB)
YOLO26s summary (fused): 122 layers, 9,466,728 parameters, 0 gradients, 20.8 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 35/35 8.8it/s 4.0s
                   all        208       2973      0.792      0.681      0.739      0.473
                player        204       2490      0.864      0.918      0.939      0.663
            goalkeeper        114        115      0.845      0.522      0.701      0.453
               referee        141        257      0.646      0.798       0.77      0.515
                  ball        111        111      0.812      0.486      0.547      0.262
Speed: 0.4ms preprocess, 7.6ms inference, 0.0ms loss, 0.4ms postprocess per image
Results saved to /content/runs/A_yolo26s_960_realval
Ultralytics 8.4.116 🚀 Python-3.12.13 torch-2.11.0+cu128 CUDA:0 (Tesla T4, 14913MiB)
YOLO26s summary (fused): 122 layers, 9,466,728 parameters, 0 gradients, 20.8 GFLOPs
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 1905.9±372.5 MB/s, size: 55.1 KB)
val: Scanning /content/football_dataset/labels/val.cache... 208 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 208/208 62.3Mit/s 0.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 13/13 2.0it/s 6.4s
                   all        208       2973      0.789      0.679      0.739      0.474
Speed: 3.7ms preprocess, 13.4ms inference, 0.0ms loss, 0.2ms postprocess per image
Results saved to /content/runs/detect/val-2
{
  "weights": "yolo26s.pt",
  "imgsz": 960,
  "epochs": 80,
  "train_minutes": 34.7,
  "best_pt": "/content/runs/A_yolo26s_960_realval/weights/best.pt",
  "split": "val",
  "mAP50": 0.7391623104564131,
  "mAP50_95": 0.47381871496141487,
  "precision_mean": 0.7886491163322376,
  "recall_mean": 0.6792201178401355,
  "per_class": {
    "player": {
      "precision": 0.86329350597995,
      "recall": 0.9180758270193154,
      "mAP50": 0.9390089139034427,
      "mAP50_95": 0.6624296755850356
    },
    "goalkeeper": {
      "precision": 0.8432667559515578,
      "recall": 0.5146527882049347,
      "mAP50": 0.7011374820237792,
      "mAP50_95": 0.45720598167063436
    },
    "referee": {
      "precision": 0.6426166877694112,
      "recall": 0.7976653696498055,
      "mAP50": 0.7699871916606443,
      "mAP50_95": 0.5143470069880054
    },
    "ball": {
      "precision": 0.8054195156280317,
      "recall": 0.4864864864864865,
      "mAP50": 0.5465156542377866,
      "mAP50_95": 0.26129219560198397
    }
  },
  "inference_ms": 13.38,
  "total_ms_per_image": 17.35,
  "fps": 57.7
}
{'weights': 'yolo26s.pt',
 'imgsz': 960,
 'epochs': 80,
 'train_minutes': 34.7,
 'best_pt': '/content/runs/A_yolo26s_960_realval/weights/best.pt',
 'split': 'val',
 'mAP50': 0.7391623104564131,
 'mAP50_95': 0.47381871496141487,
 'precision_mean': 0.7886491163322376,
 'recall_mean': 0.6792201178401355,
 'per_class': {'player': {'precision': 0.86329350597995,
   'recall': 0.9180758270193154,
   'mAP50': 0.9390089139034427,
   'mAP50_95': 0.6624296755850356},
  'goalkeeper': {'precision': 0.8432667559515578,
   'recall': 0.5146527882049347,
   'mAP50': 0.7011374820237792,
   'mAP50_95': 0.45720598167063436},
  'referee': {'precision': 0.6426166877694112,
   'recall': 0.7976653696498055,
   'mAP50': 0.7699871916606443,
   'mAP50_95': 0.5143470069880054},
  'ball': {'precision': 0.8054195156280317,
   'recall': 0.4864864864864865,
   'mAP50': 0.5465156542377866,
   'mAP50_95': 0.26129219560198397}},
 'inference_ms': 13.38,
 'total_ms_per_image': 17.35,
 'fps': 57.7}



 *** LLM ADIVSOR ANSWER ***

 Experiment A on the real validation set is complete.

Key result:
- mAP50: 0.739
- mAP50-95: 0.474
- player recall: 0.918
- goalkeeper recall: 0.515
- referee precision: 0.643
- referee recall: 0.798
- ball recall: 0.486
- FPS: 57.7

Early stopping selected epoch 12 and stopped at epoch 32.

Conclusion:
A has plateaued at 960. Do not retrain A or increase patience.

Next task:
Prepare Experiment B:
YOLO26s @ 1280.

The purpose of B is specifically to test whether higher resolution materially
improves BALL recall/localization.

Requirements:
1. Keep the exact same train/val split.
2. Do not touch the test set.
3. Use the same pretrained YOLO26s weights.
4. Use a conservative fixed batch size suitable for T4 memory.
5. Train with early stopping.
6. Compare B against A on:
   - ball recall
   - ball mAP50-95
   - referee precision
   - goalkeeper recall
   - FPS
7. Do not choose B just because overall mAP is higher.

Before running, estimate safe batch size and give the exact Colab command.
Do not modify production code.