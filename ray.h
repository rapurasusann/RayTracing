#ifndef RAYH
#define RAYH
#include "vec3.h"

class ray
{
public:
	ray() {}

	vec3 A;
	vec3 B;

	__device__ __host__ ray(const vec3 a, const vec3 b)
	{
		A = a;
		B = b;
	}

	__device__ __host__ vec3 origin() const       { return A; }
	__device__ __host__ vec3 direction() const    { return B; }//direction �����i�F���������W�j
	__device__ __host__ vec3 point_at_parameter(float t) const
	{
		return A + t*B; //�x�N�g���֐�
	}
};

#endif 

