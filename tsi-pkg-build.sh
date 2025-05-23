
set -e

#Ensure prerequisites are met as follows
echo 'updating submodule'
git submodule update --recursive --init
cd ggml-tsi-kernel/
module load tsi4 gcc/13.3.0
echo 'creating python virtual env'
python3 -m venv blob-creation
source blob-creation/bin/activate
echo 'installing mlir and python dependencies'
pip install -r /proj/rel/sw/mlir-compiler/python/requirements-common.txt
pip install /proj/rel/sw/mlir-compiler/python/mlir_external_packages-1.2.1-py3-none-any.whl
pip install onnxruntime-training

#build TSI kernels for the Tsavorite backend
#First for FPGA

echo 'creating fpga kernel'
cd fpga-kernel
cmake -B build-fpga
./create-all-kernels.sh
#The for Posix Use cases 

echo 'creating posix kernel'
cd ../posix-kernel/
./create-all-kernels.sh

#Change directory to top level llama.cpp  

cd ../../

export MLIR_SDK_VERSION=/proj/work/rel/sw/sdk-r.0.1.1
#Compile for posix with build-posix as a target folder

echo 'building llama.cp, ggml for tsavorite  and other binary for posix'
cmake -B build-posix -DGGML_TSAVORITE=ON -DGGML_TSAVORITE_TARGET=posix
cmake --build build-posix --config Release

#Compile for fpga with build-fpga as a target folder

echo 'building llama.cp, ggml for tsavorite  and other binary for fpga'
export CC="/proj/rel/sw/arm-gnu-toolchain-14.2.rel1-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-gcc"
export CXX="/proj/rel/sw/arm-gnu-toolchain-14.2.rel1-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++"
cmake -B build-fpga -DGGML_TSAVORITE=ON -DGGML_TSAVORITE_TARGET=fpga
cmake --build build-fpga --config Release


echo 'creating tar bundle for fpga'
TSI_GGML_VERSION=0.0.2
TSI_GGML_BUNDLE_INSTALL_DIR=tsi-ggml
GGML_TSI_INSTALL_DIR=ggml-tsi-kernel
TSI_GGML_RELEASE_DIR=/proj/rel/sw/ggml/
TSI_BLOB_INSTALL_DIR=$(pwd)/${GGML_TSI_INSTALL_DIR}/fpga-kernel/build-fpga

if [ -e ${TSI_GGML_BUNDLE_INSTALL_DIR} ]; then
   echo "${TSI_GGML_BUNDLE_INSTALL_DIR} exist"
else
   echo "creating ${TSI_GGML_BUNDLE_INSTALL_DIR}"
   mkdir ${TSI_GGML_BUNDLE_INSTALL_DIR}
fi
if [ -e ${TSI_GGML_BUNDLE_INSTALL_DIR}/ggml.sh ]; then
   rm -fr ${TSI_GGML_BUNDLE_INSTALL_DIR}/ggml.sh
fi

cat > ${TSI_GGML_BUNDLE_INSTALL_DIR}/ggml.sh << EOL
#!/bin/bash
export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:\$(pwd)
mkdir -p ${TSI_BLOB_INSTALL_DIR}/txe_mult
mkdir -p ${TSI_BLOB_INSTALL_DIR}/txe_add
cp blobs ${TSI_BLOB_INSTALL_DIR}/txe_mult/ -r
cp blobs ${TSI_BLOB_INSTALL_DIR}/txe_add/ -r
EOL
chmod +x ${TSI_GGML_BUNDLE_INSTALL_DIR}/ggml.sh
cp ${GGML_TSI_INSTALL_DIR}/fpga/blobs ${TSI_GGML_BUNDLE_INSTALL_DIR}/ -r
cp build-fpga/bin/llama-cli ${TSI_GGML_BUNDLE_INSTALL_DIR}/
cp build-fpga/bin/libggml*.so ${TSI_GGML_BUNDLE_INSTALL_DIR}/
cp build-fpga/bin/libllama*.so ${TSI_GGML_BUNDLE_INSTALL_DIR}/
cp build-fpga/bin/simple-backend-tsi ${TSI_GGML_BUNDLE_INSTALL_DIR}/

tar -cvzf ${TSI_GGML_BUNDLE_INSTALL_DIR}-${TSI_GGML_VERSION}.tz ${TSI_GGML_BUNDLE_INSTALL_DIR}/*

if [ "$1" == "Release" ] || [ "$1" == "release" ]
then
    cp ${TSI_GGML_BUNDLE_INSTALL_DIR}-${TSI_GGML_VERSION}.tz ${TSI_GGML_RELEASE_DIR}
fi
