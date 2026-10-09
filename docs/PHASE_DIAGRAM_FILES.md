# Phase diagram outputs

New snapshots use the model version as the folder name:

    current/phase_diagrams/
      mace-mh-1/
        phase_diagram.csv          current MLIP dataset
        history/
          phase_diagram_mlip_<hull-version>.csv
          phase_diagram_mlip_<hull-version>.json
      dft/
        phase_diagram.csv          current DFT dataset, only when data exists
        history/
          phase_diagram_dft_<hull-version>.csv
          phase_diagram_dft_<hull-version>.json

Unchanged input reuses saved diagrams and CSV files. Input record ordering is not a
new dataset. CSV requests return an existing file without running the workflow;
missing files are restored from saved snapshots, without rerunning calculations or
phase identification. New energy/structure/phase evidence changes the input checksum
and updates the current CSV while retaining immutable historical snapshots.

Legacy paths remain supported and are not moved or deleted automatically. With no
new data, the existing CSV remains the returned file. The new layout is used for
new snapshots or restoration of missing files. Existing phase-identification caches
and MLIP/DFT separation remain unchanged.
