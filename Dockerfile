# Build Parameters which can be specified via `--build-arg PARAM=VALUE`
# METALIBM_BRANCH:        branch for metalibm clone
# METALIBM_BUILD_VERSION: description of this app build version
# MWA_REPO:               repository for metalibm web app code
# MWA_BRANCH:             branch for metalibm web app clone
# ML_CUSTOM_TARGET_DIR:   directory where custom metalibm targets can be found
# HOST:                   server url
#
FROM ubuntu:18.04 AS mwa_base_deps

RUN apt-get update

RUN apt-get install -y git
RUN apt-get install -y python3 python3-setuptools libpython3-dev python3-pip
RUN apt-get install -y libmpfr-dev libmpfi-dev libfplll-dev libxml2-dev wget

FROM mwa_base_deps AS mwa_custom_deps_build

RUN apt-get install -y build-essential
# install sollya's dependency

# install sollya
WORKDIR  /home/

# sollya weekly release (which implement sollya_lib_obj_is_external_data
# contrary to sollya 7.0 release)
#RUN wget http://sollya.gforge.inria.fr/sollya-weekly-09-23-2019.tar.gz && tar -xzf sollya-weekly-09-23-2019.tar.gz
RUN wget http://sollya.gforge.inria.fr/sollya-weekly-07-06-2020.tar.gz && tar -xzf sollya-weekly*
WORKDIR /home/sollya-weekly-07-06-2020/
RUN mkdir -p /app/local/python3/ && ./configure --prefix /app/local && make -j8 && make install


# installing pythonsollya
WORKDIR /home/
ENV PATH=/app/local/bin:$PATH
ENV SOLLYA_DIR=/app/local/
ENV PREFIX=/app/local/
ENV INSTALL_OPTIONS="-t /app/local/"

# installing pythonsollya
RUN git clone https://gitlab.com/metalibm-dev/pythonsollya /home/new_pythonsollya/

# RUN apt-get install -y python3-setuptools libpython3-dev

RUN apt-get install python3-six
RUN pip3 install bigfloat

# for python 3
WORKDIR /home/new_pythonsollya/
RUN make SOLLYA_DIR=/app/local/ PREFIX=/app/local/ INSTALL_OPTIONS="-t /app/local/python3/" PYTHON=python3 PIP=pip3 install

# checking pythonsolya install
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

FROM mwa_base_deps AS mwa_metalibm_cache

COPY --from=mwa_custom_deps_build /app/local /app/local

RUN pip3 install cython

# downloading metalibm-lugdunum
ARG METALIBM_BRANCH=unknown
ENV METALIBM_BRANCH=$METALIBM_BRANCH
# useful to trigger re-build when chaning metalibm, changing
# METALIBM_BUILD_VERSION will break docker cache here
ARG METALIBM_BUILD_VERSION=unknown
ENV METALIBM_BUILD_VERSION=$METALIBM_BUILD_VERSION
ARG METALIBM_REPO=https://github.com/kalray/metalibm.git
ENV METALIBM_REPO=$METALIBM_REPO
WORKDIR /home/
RUN git clone $METALIBM_REPO -b $METALIBM_BRANCH
WORKDIR /home/metalibm/

# installing custom targets
ARG ML_CUSTOM_TARGETS_DIR=
ENV ML_CUSTOM_TARGETS_DIR=$ML_CUSTOM_TARGETS_DIR
COPY $ML_CUSTOM_TARGETS_DIR /home/metalibm/metalibm_core/targets/proprietary/


# FIXME installing metalibm python dependency manually to avoid re-installing pythonsollya
RUN pip3 install git+https://github.com/nibrunie/asmde
RUN pip3 install matplotlib
RUN apt install -y python3-yaml

ENV LD_LIBRARY_PATH=/app/local/lib/
ENV PYTHONPATH=/app/local/python3/

FROM mwa_metalibm_cache AS mwa-base-image

# setting env for experiment execution
WORKDIR /home/
ENV PYTHONPATH=/home/metalibm/:/app/local/python3/
ENV ML_SRC_DIR=/home/metalibm/

# cloning Metalibm web app
ARG MWA_REPO=https://github.com/metalibm/MetalibmWepApp.git
ENV MWA_REPO=$MWA_REPO
ARG MWA_BRANCH=unknown
ENV MWA_BRANCH=$MWA_BRANCH
RUN git clone $MWA_REPO -b $MWA_BRANCH /home/MetalibmWebApp
RUN pip3 install -r /home/MetalibmWebApp/requirements.txt

# installing ASMDE (metalibm dependency) directly
RUN pip3 install git+https://github.com/nibrunie/asmde

EXPOSE 8080
ENV PATH=/app/local/bin:$PATH

FROM mwa-base-image AS mwa-debug-image

ARG HOST="http://localhost:8080"
ENV HOST=$HOST

# CMD ["python3", "/home/MetalibmWebApp/myapp.py", "--port", "8080", "--localhost", "http://localhost:8080", "--version-info", $METALIBM_BUILD_VERSION]
CMD python3 /home/MetalibmWebApp/myapp.py --port 8080 --localhost $HOST --version-info "$METALIBM_BUILD_VERSION" --disable-log

FROM mwa-base-image AS mwa-release-image

#CMD ["python3", "/home/etalibmWebApp/myapp.py", "--port", "8080", "--localhost", "https://metalibmwebapp.appspot.com"]
