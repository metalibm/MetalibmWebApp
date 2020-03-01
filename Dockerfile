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
RUN wget http://sollya.gforge.inria.fr/sollya-weekly-09-23-2019.tar.gz && tar -xzf sollya-weekly-09-23-2019.tar.gz
WORKDIR /home/sollya-weekly-09-23-2019/
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

FROM mwa_base_deps AS mwa_metalibm_cache

COPY --from=mwa_custom_deps_build /app/local /app/local

# downloading metalibm-lugdunum
ARG METALIBM_BRANCH=unknown
ENV METALIBM_BRANCH=$METALIBM_BRANCH
# useful to trigger re-build when chaning metalibm, changing
# METALIBM_BUILD_VERSION will break docker cache here
ARG METALIBM_BUILD_VERSION=unknown
WORKDIR /home/
RUN git clone https://github.com/kalray/metalibm.git -b $METALIBM_BRANCH
WORKDIR /home/metalibm/

ENV LD_LIBRARY_PATH=/app/local/lib/
ENV PYTHONPATH=/app/local/python3/


RUN pip3 install cython

FROM mwa_metalibm_cache AS mwa-base-image

# setting env for experiment execution
WORKDIR /home/
ENV PYTHONPATH=/home/metalibm/:/app/local/python3/
ENV ML_SRC_DIR=/home/metalibm/

# cloning Metalibm web app
RUN git clone https://github.com/metalibm/MetalibmWepApp.git /home/MetalibmWebApp
RUN pip3 install -r /home/MetalibmWebApp/requirements.txt

EXPOSE 8080
ENV PATH=/app/local/bin:$PATH

FROM mwa-base-image AS mwa-debug-image

CMD ["python3", "/home/MetalibmWebApp/myapp.py", "--port", "8080", "--localhost", "http://localhost:8080"]

FROM mwa-base-image AS mwa-release-image

#CMD ["python3", "/home/etalibmWebApp/myapp.py", "--port", "8080", "--localhost", "https://metalibmwebapp.appspot.com"]
