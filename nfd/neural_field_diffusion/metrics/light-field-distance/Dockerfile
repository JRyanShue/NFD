FROM  utensils/opengl:stable

RUN apk update && apk upgrade
# dependencies
RUN apk add  \
  mesa-dev \
  freeglut-dev \
  make \
  git \
  gcc \ 
  g++ \
  wget \
  bash \
  gdb

# running
WORKDIR /usr/src/app
COPY . .

ENV DISPLAY :99 

# this is necessary to obtain the results as for the desktop app
ENV GALLIUM_DRIVER swr 

RUN cd 3DAlignment/ \
  && make \
  && make release

RUN cd LightField/ \
  && make \
  && make release 

