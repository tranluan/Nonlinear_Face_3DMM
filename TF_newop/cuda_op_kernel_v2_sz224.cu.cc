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

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include <math.h>


#define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

#define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

#define min3(a,b,c) (min(min(a,b), c))

#define max3(a,b,c) (max(max(a,b), c))

__global__ void ZbufferTriKernel(const float* s2d, const int* tri, const bool* vis, const int tri_num, const int vertex_num, int* out, float* zbuffer, int img_sz) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < tri_num - 1; i += blockDim.x * gridDim.x) {
  	if (vis[i]) {
    
	    int vt1 = tri[            i];
	    int vt2 = tri[	tri_num + i];
	    int vt3 = tri[2*tri_num + i];

	    float point1_u = s2d[             vt1];
	    float point1_v = s2d[vertex_num + vt1];

	    float point2_u = s2d[             vt2];
	    float point2_v = s2d[vertex_num + vt2];

	    float point3_u = s2d[             vt3];
	    float point3_v = s2d[vertex_num + vt3];

	    int umin =  int(ceil (double( min3(point1_u, point2_u, point3_u) )));
	    int umax =  int(floor(double( max3(point1_u, point2_u, point3_u) )));

	    int vmin =  int(ceil (double( min3(point1_v, point2_v, point3_v) )));
	    int vmax =  int(floor(double( max3(point1_v, point2_v, point3_v) )));

            float r = (s2d[2*vertex_num+vt1] + s2d[2*vertex_num+vt2] + s2d[2*vertex_num+vt3])/3;

	    
	    if (umax < img_sz && vmax < img_sz && umin >= 0 && vmin >= 0 ){
	    	for (int u = umin; u <= umax; u++){
	    		for (int v = vmin; v <= vmax; v++){
	    			
				    bool flag;
				    
				    float v0_u = point3_u - point1_u; //C - A
                                    float v0_v = point3_v - point1_v; //C - A

				    float v1_u = point2_u - point1_u; //B - A
				    float v1_v = point2_v - point1_v; //B - A

				    float v2_u = u - point1_u;
				    float v2_v = v - point1_v;

				    float dot00 = v0_u * v0_u + v0_v * v0_v;
				    float dot01 = v0_u * v1_u + v0_v * v1_v;
				    float dot02 = v0_u * v2_u + v0_v * v2_v;
				    float dot11 = v1_u * v1_u + v1_v * v1_v;
				    float dot12 = v1_u * v2_u + v1_v * v2_v;
				    
				    float inverDeno = 1 / (dot00 * dot11 - dot01 * dot01 + 1e-6);
				    float uu = (dot11 * dot02 - dot01 * dot12) * inverDeno;
				    float vv = 0;
				    if (uu < 0 or uu > 1){
				        flag = 0;
				        
				    }
				    else {
				    	vv = (dot00 * dot12 - dot01 * dot02) * inverDeno;
					    if (vv < 0 or vv > 1){
					        flag = 0;
					    }
					    else
					    {
					    	flag = uu + vv <= 1;
					    }

				    }

				    if (flag){
				    	if (zbuffer[u * img_sz + v] < r ){ // and triCpoint(np.asarray([u, v]), pt1, pt2, pt3)):
					    	zbuffer[u * img_sz + v] = r;
                                                out[u * img_sz + v ] = i;
                                        }
		                
				    }
	    		}
	    	}
	    }
	}
  }

}

__global__ void Initialize(const int tri_num, int* out, float* zbuffer, int img_sz) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < img_sz*img_sz; i += blockDim.x * gridDim.x) {
    zbuffer[i] = -INFINITY;
    out[i] = tri_num;
  }
}

__global__ void ConvertToMask(float* zbuffer, int img_sz) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < img_sz*img_sz; i += blockDim.x * gridDim.x) {
    if (zbuffer[i] == -INFINITY){
    	zbuffer[i] = 0;
    }
    else{
    	zbuffer[i] = 1;
    }
  }
}


void ZbufferTriLauncher(const float* s2d, const int* tri, const bool* vis, const int tri_num, const int vertex_num, int* out, float* zbuffer) {
	int img_sz = 224;
	Initialize<<<32, 256>>>(tri_num-1, out, zbuffer, img_sz);
	ZbufferTriKernel<<<1, 1>>>(s2d, tri, vis, tri_num, vertex_num, out, zbuffer, img_sz);
	// TODO: Make sure the correctness when process in paralell i.e ZbufferTriKernel<<<32, 256>>>(s2d, tri, vis, tri_num, vertex_num, out, zbuffer); 
	ConvertToMask<<<32, 256>>>(zbuffer, img_sz);
}

#endif
