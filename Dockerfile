#FROM ubuntu:xenial # 16.04 LTS
#FROM registry.gitlab.com/metalibm-dev/pythonsollya:ci_sollya_master
FROM ubuntu:18.04

RUN apt-get update

RUN apt-get install -y build-essential
RUN apt-get install -y git

# install sollya's dependency
RUN apt-get install -y libmpfr-dev libmpfi-dev libfplll-dev libxml2-dev

RUN apt-get install -y python3 python3-setuptools libpython3-dev python3-pip

# install sollya
WORKDIR  /home/
RUN apt-get install -y wget

# sollya weekly release (which implement sollya_lib_obj_is_external_data
# contrary to sollya 7.0 release)
RUN wget http://sollya.gforge.inria.fr/sollya-weekly-09-23-2019.tar.gz
RUN tar -xzf sollya-weekly-09-23-2019.tar.gz
WORKDIR /home/sollya-weekly-09-23-2019/
RUN mkdir -p /app/local/python3/
RUN ./configure --prefix /app/local
RUN make -j8
RUN make install


# installing pythonsollya
WORKDIR /home/
ENV PATH=/app/local/bin:$PATH
ENV SOLLYA_DIR=/app/local/
ENV PREFIX=/app/local/
ENV INSTALL_OPTIONS="-t /app/local/"

# installing pythonsollya
RUN git clone https://gitlab.com/metalibm-dev/pythonsollya /home/new_pythonsollya/

# RUN apt-get install -y python3-setuptools libpython3-dev

# for python 3
WORKDIR /home/new_pythonsollya/
RUN make SOLLYA_DIR=/app/local/ PREFIX=/app/local/ INSTALL_OPTIONS="-t /app/local/python3/" PYTHON=python3 PIP=pip3 install

# checking pythonsolya install
RUN apt-get install python3-six
RUN LD_LIBRARY_PATH="/app/local/lib" python3 -c "import sollya"

# installing gappa
WORKDIR /home/
RUN apt-get install -y libboost-dev
RUN wget https://gforge.inria.fr/frs/download.php/file/37624/gappa-1.3.3.tar.gz
RUN tar -xzf gappa-1.3.3.tar.gz
WORKDIR /home/gappa-1.3.3/
RUN ./configure --prefix=/app/local/
RUN ./remake
RUN ./remake install

# downloading metalibm-lugdunum
WORKDIR /home/
RUN git clone https://github.com/kalray/metalibm.git -b new_vector_lib
WORKDIR /home/metalibm/

ENV LD_LIBRARY_PATH=/app/local/lib/
ENV PYTHONPATH=/app/local/python3/
RUN PYTHONPATH=$PWD:$PYTHONPATH  ML_SRC_DIR=$PWD python3 valid/soft_unit_test.py


RUN pip3 install cython

# setting env for experiment execution
WORKDIR /home/
ENV PYTHONPATH=/home/metalibm/:/app/local/python3/
ENV ML_SRC_DIR=/home/metalibm/

# cloning Metalibm web app
RUN git clone https://github.com/metalibm/MetalibmWepApp.git /home/MetalibmWebApp
RUN pip3 install -r /home/MetalibmWebApp/requirements.txt

EXPOSE 8080

RUN echo "#! /bin/sh\ncd /home/MetalibmWebApp/ && python3 myapp.py" > /app/local/bin/launch_app
RUN chmod 777 /app/local/bin/launch_app
