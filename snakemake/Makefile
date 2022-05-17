envs.full:
	mamba env create -f environment.yml
envs.simple:
	mamba install -c conda-forge -c bioconda snakemake
run:
	snakemake --cores all -F
run_with_singularity:
	snakemake --cores all -F --use-singularity
dry.run:
	snakemake --cores all -F --dry-run
report:
	snakemake --report report.html