#!/bin/bash

# Stop on error
set -e

git clone https://github.com/Biwei-Huang/Generalized-Score-Functions-for-Causal-Discovery.git ./other_methods/GES
git clone https://github.com/DAMO-DI-ML/AAAI2022-HCM.git ./other_methods/AAAI2022-HCM-main
mv ./other_methods/lgbm_cvmodel.py ./other_methods/AAAI2022-HCM-main/lgbm_cvmodel.py

if command -v curl &> /dev/null; then
    curl -L https://eda.rg.cispa.io/prj/crack/crack-v20190110.zip | ditto -xk - ./other_methods/
else
    wget https://eda.rg.cispa.io/prj/crack/crack-v20190110.zip
    unzip crack-v20190110.zip -d ./other_methods/
    rm crack-v20190110.zip
fi

make -C ./other_methods/crack/code/
make -C ./other_methods/crack/code/ cp
