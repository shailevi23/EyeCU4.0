# SoccerTrack v2 audit -- evidence location

The audit of record is version-controlled at:

    experiments/soccertrack_audit/

    reports/AUDIT_SUMMARY.json     inventory, GSR/BAS/RAW findings, calibration,
                                   diversity, leakage, licence, scorecard
    reports/gsr_scan.json          per-half category counts for all 20 halves
    reports/gsr_geometry.json      box geometry and track persistence
    reports/components.json        BAS, squad metadata, per-match camera geometry
    reports/calibration_128058.json  two independent transform routes, verified
    reports/alignment_128058.json  the video/annotation alignment search
    contact_sheets/                calibration overlay and alignment renders

Not duplicated here, for the same reason as the Roboflow audit.

This directory holds the data itself:

    ../gsr/       20 GSR JSON (55 GB)
    ../bas/       ball action spotting events
    ../raw/       calibration, tracking XML, squad metadata, and the EMPTY mot zip
    ../videos/    match 128058, both halves
    ../public_repo/  github.com/AtomScott/SoccerTrack-v2 checkout (code MIT)
    ../manifests/ archive hashes

MOT acquisition is CLOSED. See the SOCCERTRACK_MOT record in the master registry.
