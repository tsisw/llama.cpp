
set -e

#Ensure prerequisites are met as follows
git submodule update --recursive --init
cd ggml-tsi-kernel/
module load tsi4 gcc/13.3.0
python3 -m venv blob-creation
source blob-creation/bin/activate
pip install -r /proj/rel/sw/mlir-compiler/python/requirements-common.txt
pip install /proj/rel/sw/mlir-compiler/python/mlir_external_packages-1.2.1-py3-none-any.whl
pip install onnxruntime-training

#build TSI kernels for the Tsavorite backend
#First for FPGA

cd fpga-kernel
cmake -B build-fpga
./create-all-kernels.sh
#The for Posix Use cases 

cd ../posix-kernel/
./create-all-kernels.sh

#Change directory to top level llama.cpp  

cd ../../

#Compile for posix with build-posix as a target folder

cmake -B build-posix -DGGML_TSAVORITE=ON -DGGML_TSAVORITE_TARGET=posix
cmake --build build-posix --config Release

#Compile for fpga with build-fpga as a target folder

cmake -B build-fpga -DGGML_TSAVORITE=ON -DGGML_TSAVORITE_TARGET=fpga -DCMAKE_C_COMPILER="/proj/rel/sw/arm-gnu-toolchain-14.2.rel1-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-gcc" -DCMAKE_CXX_COMPILER="/proj/rel/sw/arm-gnu-toolchain-14.2.rel1-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++"
cmake --build build-fpga --config Release


TSI_BNDL_INSTALL_DIR=tsi-ggml-bndl
GGML_TSI_INSTALL_DIR=ggml-tsi-kernel
if [ -e ${TSI_BNDL_INSTALL_DIR} ]; then
   echo "tsi-ggml-bndl exist"
else
   echo "creating tsi-ggml-bndl"
   mkdir tsi-ggml-bndl
fi

cp ${GGML_TSI_INSTALL_DIR}/fpga/blobs ${TSI_BNDL_INSTALL_DIR}/ -r
cp build-fpga/bin/llama-cli ${TSI_BNDL_INSTALL_DIR}/ 
cp build-fpga/bin/libggml*.so ${TSI_BNDL_INSTALL_DIR}/ 
cp build-fpga/bin/libllama*.so ${TSI_BNDL_INSTALL_DIR}/ 
cp build-fpga/bin/simple-backend-tsi ${TSI_BNDL_INSTALL_DIR}/

tar -cvzf tsi-ggml.gz ${TSI_BNDL_INSTALL_DIR}
