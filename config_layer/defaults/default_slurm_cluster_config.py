"""Site-specific Slurm profiles supplied for the layered-oxide workflow."""


def default_slurm_cluster_config() -> dict:
    """Return editable profiles without submitting any cluster work."""
    return {
        "mlip_gpu": {
            "stages": ["relax_and_feature", "deep_search"],
            "slurm_options": {
                "nodes": 1,
                "ntasks": 2,
                "ntasks-per-node": 2,
                "mem": "32G",
                "gres": "gpu:1",
                "partition": "v100m3",
                "output": "%A_%a.out",
                "error": "%A_%a.err",
            },
            "shell_preamble": [
                'export cuda_path="/data/app/cuda/cuda-12.8"',
                'export PATH="${cuda_path}/bin:${PATH}"',
                'export LD_LIBRARY_PATH="${cuda_path}/lib64:${LD_LIBRARY_PATH:-}"',
                'export CUDA_HOME="${cuda_path}"',
                "source /data/app/anaconda3/2024.10-1/bin/activate",
                "conda activate mace",
            ],
        },
        "vasp_atomate": {
            "stages": ["dft_single_point", "dft_relax"],
            "slurm_options": {
                "nodes": 1,
                "ntasks": 1,
                "ntasks-per-node": 1,
                "partition": "v100m3",
                "gres": "gpu:1",
                "output": "%j.out",
                "error": "%j.err",
            },
            "shell_preamble": [
                "module load gcc/12.2.0",
                'export vasp_path="/data/app/vasp/6.5.1-nvhpc"',
                "module use /data/app/nvhpc/22.5_cuda12.9/modulefiles/",
                "module load nvhpc-hpcx fftw/3.3.10-nvhpc",
                'export PATH="${vasp_path}/bin:${PATH}"',
                "ulimit -s unlimited",
                "ulimit -l unlimited",
            ],
            "vasp_command": "mpirun -np ${SLURM_NPROCS} vasp_std",
        },
    }
