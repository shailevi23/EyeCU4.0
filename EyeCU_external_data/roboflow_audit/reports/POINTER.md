# Roboflow six-source audit -- evidence location

The audit of record is version-controlled at:

    experiments/external_data_audit/

    raw/SOURCES.json          the six ZIPs: workspace, project, version, licence, sha256
    reports/                  inventory, class map, annotation quality, duplicates,
                              leakage, orientation, integrity, AUDIT_SUMMARY
    contact_sheets/           32 rendered sheets
    candidate_index/          per-image KEEP/REVIEW/EXCLUDE status and reasons

It is NOT duplicated here. The reports are small text and belong in git; copying
them beside the ZIPs would create a second copy that can drift from the first.

This directory holds only what is specific to the raw ZIPs:

    ../raw_zips/              the six immutable archives
    ../manifests/raw_zip_hashes.json   hashes re-verified after consolidation

The audit is COMPLETE and must not be re-run except to validate a newly
discovered duplicate source.
