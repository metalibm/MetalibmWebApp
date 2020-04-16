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


# cgpe dependencies
RUN apt-get install -y subversion autoconf libtool pkg-config

# installing cgpe
ENV PATH=/app/local/bin:$PATH
WORKDIR /home/cgpe/
RUN svn checkout svn://scm.gforge.inria.fr/svnroot/cgpe/
WORKDIR /home/cgpe/cgpe/trunk/
RUN . ./autogen.sh && ./configure --prefix=/app/local
RUN make && make install

RUN pip3 install cython

# installing pythoncgpe
WORKDIR /home/
RUN git clone https://scm.gforge.inria.fr/anonscm/git/metalibm/pythoncgpe.git
WORKDIR /home/pythoncgpe/
ENV CFLAGS="-I/app/local/include/cgpe/ -I/app/local/include/cgpe/analyzers/ -I/usr/include/libxml2/ -L/app/local/lib/"
ENV CPPFLAGS="-L/app/local/lib/cgpe/"
RUN python3 setup.py build
RUN pip3 install -t /app/local/python3 .

FROM mwa_base_deps AS mwa_metalibm_cache

COPY --from=mwa_custom_deps_build /app/local /app/local

# downloading metalibm-lugdunum
ARG METALIBM_BRANCH=master
ENV METALIBM_BRANCH=$METALIBM_BRANCH
# useful to trigger re-build when chaning metalibm, changing
# METALIBM_BUILD_VERSION will break docker cache here
ARG METALIBM_BUILD_VERSION="generic implementpoly"
ENV METALIBM_BUILD_VERSION=$METALIBM_BUILD_VERSION
WORKDIR /home/
RUN git clone https://github.com/kalray/metalibm.git -b $METALIBM_BRANCH
WORKDIR /home/metalibm/

ENV LD_LIBRARY_PATH=/app/local/lib/
ENV PYTHONPATH=/app/local/python3/


FROM mwa_metalibm_cache AS mwa-gp-base-image




ENV LD_LIBRARY_PATH=/app/local/lib/cgpe/:/app/local/lib/

# metalibm lutetia needs Python.h
RUN apt-get install libpython3-dev

# downloading and installing metalibm-lutetia
WORKDIR /home/
RUN git clone https://scm.gforge.inria.fr/anonscm/git/metalibm/metalibm-lutetia.git -b genericimplementpoly ./metalibm_lutetia
WORKDIR /home/metalibm_lutetia/
ADD ./python3_config.patch /home/metalibm_lutetia/
RUN git apply ./python3_config.patch
ENV CFLAGS="-I/app/local/include/ -L/app/local/lib/"
ENV CPPFLAGS="-L/app/local/lib/"
RUN make all

# moving generic implementpoly repo
ADD ./tools/*.py /home/genericimplementpoly/tools/
ADD ./tools/*.sol /home/genericimplementpoly/tools/

# setting env for experiment execution
WORKDIR /home/
ENV PYTHONPATH=/home/metalibm/:/home/metalibm_lutetia/:/home/genericimplementpoly/tools/:/app/local/python3/
ENV ML_SRC_DIR=/home/metalibm/

ENV METALIBM_CFLAGS="-mfma -mavx2 -I/home/metalibm/metalibm_core "
ENV GENERICIMPLEMENTPOLY_ROOT=/home/genericimplementpoly/tools/
ENV METALIBM_LUTETIA_BIN=/home/metalibm_lutetia/metalibm

# cloning Metalibm web app
ARG MWA_BRANCH=master
ENV MWA_BRANCH=$MWA_BRANCH
#RUN git clone https://github.com/metalibm/MetalibmWepApp.git -b $MWA_BRANCH /home/MetalibmWebApp 
ADD ./requirements.txt /home/MetalibmWebApp/
RUN pip3 install -r /home/MetalibmWebApp/requirements.txt

ADD ./myapp.py /home/MetalibmWebApp/
ADD ./main.xhtml /home/MetalibmWebApp/
ADD ./stats.xhtml /home/MetalibmWebApp/
ADD ./public/mwa.js /home/MetalibmWebApp/public/

EXPOSE 8080
ENV PATH=/app/local/bin:$PATH

FROM mwa-gp-base-image AS mwa-gp-debug-image

#CMD ["python3", "/home/MetalibmWebApp/myapp.py", "--port", "8080", "--localhost", "http://localhost:8080", "--version-info", $METALIBM_BUILD_VERSION]
CMD python3 /home/MetalibmWebApp/myapp.py --port 8080 --localhost "http://localhost:8080" --version-info "$METALIBM_BUILD_VERSION" --disable-log --ml-lutetia-dir /home/metalibm_lutetia

FROM mwa-gp-base-image AS mwa-gp-release-image

#CMD ["python3", "/home/etalibmWebApp/myapp.py", "--port", "8080", "--localhost", "https://metalibmwebapp.appspot.com"]
