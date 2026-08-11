Drop CVAT exports here, one per sequence, named exactly:

    austin_fc_vs__club_tijuana_284.xml
    bayern_munich_3-1_chelsea_228.xml
    women_1_239.xml
    youth_premier_league_1133.xml

Format must be "CVAT for video 1.1". Anything else (CVAT for images, MOT)
either loses identity or loses the role label; the importer detects a
shape-only export and refuses.

Then run:  python tools/import_tracking_gt_cvat.py

See docs/guides/IDENTITY_GT_ANNOTATION.md for the full workflow.
