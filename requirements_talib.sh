#!/bin/bash
# This link needs to be updated for script to work
wget https://nav.dl.sourceforge.net/project/ta-lib/ta-lib/0.4.0/ta-lib-0.4.0-src.tar.gz &&
tar -xzvf ta-lib-0.4.0-src.tar.gz &&
cd ta-lib/ &&
./configure --prefix=/usr &&
make &&
sudo make install