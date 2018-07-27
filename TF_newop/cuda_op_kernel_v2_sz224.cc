/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"


using namespace tensorflow;  // NOLINT(build/namespaces)

REGISTER_OP("ZbufferTriV2Sz224")
    .Input("s2d: float")
    .Input("tri: int32")
    .Input("vis: bool")
    .Output("output: int32")
    .Output("zbuffer: float")
    .Doc(R"doc(
)doc");



void ZbufferTriLauncher(const float* s2d, const int* tri, const bool* vis, const int tri_num, const int vertex_num, int* out, float* zbuffer);

class ZbufferTriOp : public OpKernel {
 public:
  explicit ZbufferTriOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    int img_sz = 224;
    // Grab the input tensors
    const Tensor& s2d_tensor = context->input(0);
    auto s2d = s2d_tensor.flat<float>();

    const Tensor& tri_tensor = context->input(1);
    auto tri = tri_tensor.flat<int32>();

    const Tensor& vis_tensor = context->input(2);
    auto vis = vis_tensor.flat<bool>();

    // Create an output tensor
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({img_sz, img_sz}), &output_tensor));
    auto output = output_tensor->template flat<int32>();

    
    Tensor* zbuffer_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({img_sz, img_sz}), &zbuffer_tensor));
    auto zbuffer = zbuffer_tensor->template flat<float>();

    // Set all but the first element of the output tensor to 0.
    const int tri_num     = tri_tensor.shape().dim_size(1);
    const int vertex_num  = s2d_tensor.shape().dim_size(1);
    // Call the cuda kernel launcher
    ZbufferTriLauncher(s2d.data(), tri.data(), vis.data(), tri_num, vertex_num, output.data(), zbuffer.data());
  }
};

REGISTER_KERNEL_BUILDER(Name("ZbufferTriV2Sz224").Device(DEVICE_GPU), ZbufferTriOp);
