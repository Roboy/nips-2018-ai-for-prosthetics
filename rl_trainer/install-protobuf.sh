#!/usr/bin/env bash

# protoc compiler
wget https://github.com/protocolbuffers/protobuf/releases/download/v3.5.0/protoc-3.5.0-linux-x86_64.zip
unzip protoc-3.5.0-linux-x86_64.zip -d "/home/$USER"
rm protoc-3.5.0-linux-x86_64.zip

# protoc python
wget https://github.com/protocolbuffers/protobuf/releases/download/v3.6.0/protobuf-python-3.6.0.zip
unzip protobuf-python-3.6.0.zip
rm protobuf-python-3.6.0.zip

cd protobuf-3.6.0/python/
python setup.py build
python setup.py test
python setup.py install
cd ../../
rm -rf protobuf-3.6.0/