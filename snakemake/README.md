# Demo using snakemake to run a python container with a specific script across folder.

Often times when analysing biological images there is a need to run them through multiple processing steps and at scale.
When doing this linearly in something like ImageJ this can be a lengthy process and individual tweaks to each processing step can mean running the entire dataseries again.
With workflow managers we get a set of tools for creating
`findable`, `accessible`, `interoperable`, and `reusable` analysis routines.
Typically workflow managers are lists of "rules" with input and outputs for data-flow.
This provides a level of concurrency and redundancy for re-running analyses with tweaks and adjustments.
Workflow managers also make scaling analysis up convenient by treating inputs as parallel operations where possible.
The more advanced workflow managers can then deploy to cloud infrastructure for big-data analysis

## Snakemake

This demo will use Snakemake as it's workflow manager, a popular alternative is Nextflow.
Snakemake is written in- and can interpret raw Python and so is more fitting to this course.

    conda env update --file environment.yml

### Cool_script

We start by generating a file called `cool_script.py`
In the example provided cool_script.py takes in a `tif` image and scales it to a dynamic range of `0-1` and then outputs the image as a `zarr`.
In out `Snakefile` (usually kept in the root of the directory) we can the define the inputs and outputs of the rule we intend to make for running `cool_scipt.py`

    FILE = "in/Stardist/Test - Images/cell migration R1 - Position 58_XY1562686154_Z0_T00_C1-image76.tif"
    FILE_OUT = "out/out.zarr"

We can then define a rule for running our `cool_script.py`

    rule scale_image_down_make_zarr:

Then we define the input and outputs using the variables above

        input:
            FILE
        output:
            directory(FILE_OUT)

N.b. that the output of the script is a directory and must be defined as such.

finally we define the conda environment file to use:

        conda:
            "environment.yml"

and the command to run the `cool_script.py`

        shell:
            "python cool_script.py '{input}' '{output}'"

The `{}` variables in this case refer to the `input:` and `output:`  lines defined above, once the Snakefile is run these variables will be substituted.

the environment file in this case is as follows:

    name: snakemake
        channels:
            - conda-forge
            - bioconda
        dependencies:
            - python=3.8   # or 2.7
            - pims
            - ipython
            - pip
            - numpy
            - scikit-learn
            - snakemake
            - pip:
                - ome-zarr

We can now run this rule with:
    
    $ snakemake --until scale_image_down_make_zarr --cores all -F

Where the `--until` flag refers to the rule we want to run (`scale_image_down_make_zarr`), the ` --cores` flag defines how many CPU cores we want to use for analysis (`all` in this case) and the `-F` flag forces the pipeline to run.

This produces an ASCII report:

    snakemake ❯ snakemake --until scale_image_down_make_zarr --cores all -F
    Building DAG of jobs...
    Using shell: /usr/bin/bash
    Provided cores: 16
    Rules claiming more threads will be scaled down.
    Conda environments: ignored
    Singularity containers: ignored
    Job stats:
    job                           count    min threads    max threads
    --------------------------  -------  -------------  -------------
    scale_image_down_make_zarr        1              1              1
    total                             1              1              1

    Select jobs to execute...

    [Wed Jun 30 10:30:54 2021]
    rule scale_image_down_make_zarr:
        input: in/Stardist/Test - Images/cell migration R1 - Position 58_XY1562686154_Z0_T00_C1-image76.tif
        output: out/out.zarr
        jobid: 2
        resources: tmpdir=/tmp

    in/Stardist/Test - Images/cell migration R1 - Position 58_XY1562686154_Z0_T00_C1-image76.tif
    [Wed Jun 30 10:30:55 2021]
    Finished job 2.
    1 of 4 steps (25%) done
    Complete log: /home/ctr26/+projects/2021_summer_course/day_1/snakemake/.snakemake/log/2021-06-30T103054.529500.snakemake.log


### Cool_script_2

We have now decided that we want to produce a small png file for debugging our analysis workflow so we have created `cool_script_2.py`.
We then define another rule in out Snakefile:

    rule make_a_nice_png:
        input:
            FILE_OUT
        output:
            FILE_OUT_PNG
        conda:
            "environment.yml"
        shell:
            "python cool_script_2.py '{input}' '{output}'"

This time the `input:` line refers to the `output:` of the rule `rule scale_image_down_make_zarr`. 
We make sure to define the png file:

    FILE_OUT_PNG = "out/out.png"

Snakemake will then figure out the network-graph of what outputs depend on what inputs.
We run this with the following command:

    $ snakemake --until make_a_nice_png --cores all -F

Producing the following report:

    snakemake ❯  snakemake --until make_a_nice_png --cores all -F
    Building DAG of jobs...
    Using shell: /usr/bin/bash
    Provided cores: 16
    Rules claiming more threads will be scaled down.
    Conda environments: ignored
    Singularity containers: ignored
    Job stats:
    job                           count    min threads    max threads
    --------------------------  -------  -------------  -------------
    make_a_nice_png                   1              1              1
    scale_image_down_make_zarr        1              1              1
    total                             2              1              1

    Select jobs to execute...

    [Wed Jun 30 10:34:19 2021]
    rule scale_image_down_make_zarr:
        input: in/Stardist/Test - Images/cell migration R1 - Position 58_XY1562686154_Z0_T00_C1-image76.tif
        output: out/out.zarr
        jobid: 1
        resources: tmpdir=/tmp

    in/Stardist/Test - Images/cell migration R1 - Position 58_XY1562686154_Z0_T00_C1-image76.tif
    [Wed Jun 30 10:34:20 2021]
    Finished job 1.
    1 of 4 steps (25%) done
    Select jobs to execute...

    [Wed Jun 30 10:34:20 2021]
    rule make_a_nice_png:
        input: out/out.zarr
        output: out/out.png
        jobid: 0
        resources: tmpdir=/tmp

    out/out.zarr
    [Wed Jun 30 10:34:20 2021]
    Finished job 0.
    2 of 4 steps (50%) done
    Complete log: /home/ctr26/+projects/2021_summer_course/day_1/snakemake/.snakemake/log/2021-06-30T103419.682546.snakemake.log

### microscopeimagequality

We've now found a paper that talks about using machine learning for reading out focus quality of images using machine learning.
The paper is a bit old and we're having difficulty getting it to work in a conda environment because there are several steps to building the application and running it:

    https://github.com/google/microscopeimagequality

Fortunately we have an idea on how to containerise their code so that it can run using Docker.
So we write a Dockerfile using their build instructions:

    # Correct python version for tensorflow=1.15.4
    FROM python:3.7 
    # Clone their repo
    RUN git clone -b main https://github.com/google/ microscopeimagequality.git
    # Switch to the cloned dir
    WORKDIR "./microscopeimagequality"
    # Install the package dependencies
    RUN pip3 install --editable .
    # Use their code to download their pretrained model
    RUN microscopeimagequality download 
    # Add microscopeimagequality to the environment
    ENV PYTHONPATH="${PYTHONPATH}:~/microscopeimagequality"
    # Add microscopeimagequality as an entry point
    ENTRYPOINT ["microscopeimagequality"]

We can now build this Dockerfile using:

    $ docker build ./microscopeimagequality -t ctr26/microscopeimagequality

The -t flag tags the image with something human readable.

This image is then ready to be pushed to the internet (dockerhub in this case):

    $ docker push ctr26/microscopeimagequality

To use this container as our environment in our workflow we simply replace the `conda:` line with a container line as follows:

        container:
            "docker://ctr26/microscopeimagequality:latest"

This will pull the docker image from Dockerhub and then convert it to an image format Snakemake is comfortable with (Singularity)

We then define a shell script to run the operation as before:

        shell:
            "microscopeimagequality predict --output '{output}' '{input}'"

And define the inputs and outputs as before:

        input:
            FILE
        output:
            directory(FILE_OUT_QUALITY)

Making sure to define the output folder:

    FILE_OUT_QUALITY = "out/quality"

Now, to run this rule we need to tell Snakemake to use the container we've creates by adding the `--use-singularity` flag:

    snakemake --until docker_leg  --cores all -F --use-singularity

### All

Finally we define a special rule called `all` which encompasses the entire workflow, whereby the outputs of the rules are inputs of `all`

    rule all:
        input:
            FILE_OUT_PNG,FILE_OUT_QUALITY,FILE_OUT

We can then run the entire workflow as follows:

    $ snakemake  --cores all -F --use-singularity

Snakemake is kind enough to even generate nice HTML reports for us:

    $ snakemake --report report.html

### Multiple files



## Exercises:

### cool_script.py

Update `cool_script.py` output a dynamic range of 255 instead of 1

Update `cool_script.py` output to subtract the median value of the image and then rescale to 0-1 (lazy noise removal)

### cool_script_2.py

Update `cool_script_2.py` to resize the image to 21:9 (cinemascope)
Bonus points for adding black bars top and bottom

### Snakemake

Run `microscopeimagequality` after `cool_script_1.py` and `cool_script_2.py`
