## ----------- Test with TF v1.8

TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

nvcc -std=c++11 -c -o cuda_op_kernel_v2_sz224.cu.o cuda_op_kernel_v2_sz224.cu.cc ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -D_MWAITXINTRIN_H_INCLUDED

g++ -std=c++11 -shared -o cuda_op_kernel_v2_sz224.so cuda_op_kernel_v2_sz224.cc cuda_op_kernel_v2_sz224.cu.o ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -D_GLIBCXX_USE_CXX11_ABI=0 -L /usr/local/cuda/lib64/


## ----------- Tested with TF v1.3
#TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')

#nvcc -std=c++11 -c -o cuda_op_kernel_v2_sz224.cu.o cuda_op_kernel_v2_sz224.cu.cc -I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -D_MWAITXINTRIN_H_INCLUDED

#g++ -std=c++11 -shared -o cuda_op_kernel_v2_sz224.so cuda_op_kernel_v2_sz224.cc cuda_op_kernel_v2_sz224.cu.o -I $TF_INC -fPIC -lcudart -D_GLIBCXX_USE_CXX11_ABI=0
