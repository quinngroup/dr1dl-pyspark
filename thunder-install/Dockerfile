# This Dockerfile script is to provide a template for how to install and
# get thunder-project up and running.

FROM ubuntu:14.04.3
MAINTAINER Quinn Research Group "spq@uga.edu"

USER root
ENV DEBIAN_FRONTEND noninteractive

# Basic updates, don't worry about these.
RUN apt-get update
RUN apt-get install -y wget rsync build-essential openjdk-7-jdk vim git \
    libblas-dev liblapack-dev

# Download and install Spark. Remember: it has to be built against Hadoop 1.x,
# NOT Hadoop 2.x. This is the unfortunate reality, though if you want bonus
# points, see if you can port thunder-python to Hadoop 2.x? :)
RUN wget http://d3kbcqa49mib13.cloudfront.net/spark-1.5.2-bin-hadoop1.tgz && \
    tar zxvf spark-1.5.2-bin-hadoop1.tgz && rm spark-1.5.2-bin-hadoop1.tgz
RUN mkdir /opt/spark && mv spark-1.5.2-bin-hadoop1 /opt/spark/
ENV SPARK_HOME /opt/spark/spark-1.5.2-bin-hadoop1

# Now, download the Anaconda installer for Python 2.7.
RUN wget https://3230d63b5fc54e62148e-c95ac804525aac4b6dba79b00b39d1d3.ssl.cf1.rackcdn.com/Anaconda2-2.4.1-Linux-x86_64.sh
RUN /bin/bash Anaconda2-2.4.1-Linux-x86_64.sh -b -p /opt/conda && \
    rm Anaconda2-2.4.1-Linux-x86_64.sh
ENV PATH /opt/conda/bin:$PATH
RUN conda update -y --all

# Install Thunder.
RUN pip install thunder-python

# Make IPython the default thunder console.
ENV IPYTHON 1

# All done!
CMD ["/bin/bash"]

# At this point, if you want to download and run the PySpark dictionary learning
# framework, you'll need to either generate a new SSH key and add it to the 
# authorized list of keys in the github repository (would not advise this route),
# or import your own SSH key into the running container (I advise this route).
