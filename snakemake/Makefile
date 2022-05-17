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
	
run.scale_image_down_make_zarr:
	snakemake --until scale_image_down_make_zarr --cores all -F

run.scale_image_down_make_zarr.conda:
	snakemake --until scale_image_down_make_zarr --cores all -F --use-conda