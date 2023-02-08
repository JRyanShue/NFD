#!/usr/bin/env bash

cd ./lfd/3DAlignment/
make
make release
cd ../..

cd ./lfd/LightField/
make
make release
cd ../..
