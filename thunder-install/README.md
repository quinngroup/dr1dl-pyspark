# thunder-install

This is a small Dockerfile that performs a barebones installation and configuration of [thunder-python](http://thunder-project.org/). It is provided as a general reference for installing thunder, as well as a specific resource for quickly setting up a development / deployment environment.

## Dependencies

There are a few dependencies to installing thunder-python. You can read about these on [thunder's installation docs](http://thunder-project.org/thunder/docs/install_local.html), but in particular:

 1. Python 2.6+ (currently **not** Python 3, [see this ticket](https://github.com/thunder-project/thunder/issues/209))
 2. A standard Python scientific stack: SciPy, NumPy, etc. Recommend [installing Anaconda](https://store.continuum.io/cshop/anaconda/) to satisfy these dependencies.
 2. Spark 1.5.x, **built against Hadoop 1.x.**

## Installing Docker and building thunder-install container

[Docker](https://www.docker.com/) is an open source container platform that can be thought of as a "lightweight" virtual machine. Effectively, you can write bash scripts that lay out the complete configuration for an app and execute the script, constructing a fully-replicable environment to exact specifications. Other people can then build the same environment to run your app.

[Installation of the Docker engine](https://docs.docker.com/) will vary depending on your operating system, but Docker has excellent documentation for all cases.

Once Docker is installed on your system, it should be a simple matter of running a single command to start executing this Dockerfile:

    docker build path/to/thunder-install

Docker will look for this `Dockerfile` and run the instructions therein, constructing your environment.
